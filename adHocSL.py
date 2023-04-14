
import models.ResNet as my_resnet
import datasets.cifar_data as cifar_data

import torch
import torchvision
import torchvision.transforms as transforms
import math
import numpy as np
import torch
import time
from torch.autograd import Function
from torch.optim import Adam
import torch.nn as nn
import copy

class define_training_param:
    lr=0.001
    epoch_num = 20
    batch_size = 32
    num_out = 10

class DataOwner:
    num_of_samples = 1

    def __init__(self, id, model_part_a, model_part_b, model_part_c, training_par):
        self.id = id
        self.model_part_a = model_part_a
        self.model_part_b = model_part_b
        self.model_part_c = model_part_c
        self.training_par = training_par

        # define a different optimizer for each model part
        self.optimizer_a = Adam(self.model_part_a.parameters(), lr=training_par.lr)  # NOTE: adapt accordingly
        self.optimizer_b = Adam(self.model_part_b.parameters(), lr=training_par.lr)
        self.optimizer_c = Adam(self.model_part_c.parameters(), lr=training_par.lr)

        self.criterion = nn.CrossEntropyLoss()                                  # NOTE: adapt accordingly



class AdHocSL:
    data_owners = []

    def __init__(self, pointa, pointb, num_dataowners, model_name):  # TODO: pass the model definition
        self.pointa = pointa
        self.pointb = pointb
        self.training_par = define_training_param()

        self.g_model_part_a = my_resnet.get_resnet18(self.training_par.num_out, -1, self.pointa)
        self.g_model_part_b = my_resnet.get_resnet18(self.training_par.num_out, self.pointa, self.pointb)
        self.g_model_part_c = my_resnet.get_resnet18(self.training_par.num_out, self.pointb, -1)
        

        for i in range(num_dataowners):
            model_part_a = my_resnet.get_resnet18(self.training_par.num_out, -1, self.pointa)
            model_part_b = my_resnet.get_resnet18(self.training_par.num_out, self.pointa, self.pointb)
            model_part_c = my_resnet.get_resnet18(self.training_par.num_out, self.pointb, -1)
            
            model_part_a.load_state_dict(copy.deepcopy(self.g_model_part_a.state_dict()))
            model_part_b.load_state_dict(copy.deepcopy(self.g_model_part_b.state_dict()))
            model_part_c.load_state_dict(copy.deepcopy(self.g_model_part_c.state_dict()))
            
            self.data_owners.append(DataOwner(i, model_part_a, model_part_b, model_part_c, self.training_par))


    def local_update(self, d_id, input, label):
        self.data_owners[d_id-1].optimizer_a.zero_grad()
        self.data_owners[d_id-1].optimizer_b.zero_grad()
        self.data_owners[d_id-1].optimizer_c.zero_grad()

        # start forward propagation

        out_a = self.data_owners[d_id-1].model_part_a(input)
        det_out_a = out_a.clone().detach().requires_grad_(True)


        out_b = self.data_owners[d_id-1].model_part_b(det_out_a)
        det_out_b = out_b.clone().detach().requires_grad_(True)

        out_c = self.data_owners[d_id-1].model_part_c(det_out_b)
        loss = self.data_owners[d_id-1].criterion(out_c, label)  # calculates cross-entropy loss
        
        # start backward propagation

        loss.backward() 

        loss = loss.item()
        acc = cifar_data.categorical_accuracy(out_c, label)
        acc = acc.item()

        self.data_owners[d_id-1].optimizer_c.step()

        grad_b = det_out_b.grad.clone().detach()
        out_b.backward(grad_b)
        self.data_owners[d_id-1].optimizer_b.step()

        grad_a = det_out_a.grad.clone().detach()
        out_a.backward(grad_a)
        self.data_owners[d_id-1].optimizer_a.step()

        return (loss, acc)


    def adHoc_update(self, source_id, destination_id, input, label):
        self.data_owners[source_id-1].optimizer_a.zero_grad()
        self.data_owners[source_id-1].optimizer_b.zero_grad()
        self.data_owners[source_id-1].optimizer_c.zero_grad()

        self.data_owners[destination_id-1].optimizer_a.zero_grad()
        self.data_owners[destination_id-1].optimizer_b.zero_grad()
        self.data_owners[destination_id-1].optimizer_c.zero_grad()

        # start forward propagation

        out_a = self.data_owners[source_id-1].model_part_a(input)
        det_out_a = out_a.clone().detach().requires_grad_(True)

        # TRANSFER

        out_b = self.data_owners[destination_id-1].model_part_b(det_out_a)
        det_out_b = out_b.clone().detach().requires_grad_(True)

        # TRANSFER

        out_c = self.data_owners[source_id-1].model_part_c(det_out_b)
        
        loss = self.data_owners[source_id-1].criterion(out_c, label)  # calculates cross-entropy loss
        
        # start backward propagation

        loss.backward() 

        loss = loss.item()
        acc = cifar_data.categorical_accuracy(out_c, label)
        acc = acc.item()

        self.data_owners[source_id-1].optimizer_c.step()

        grad_b = det_out_b.grad.clone().detach()

        # TRANSFER

        out_b.backward(grad_b)
        self.data_owners[destination_id-1].optimizer_b.step()

        grad_a = det_out_a.grad.clone().detach()

        # TRANSFER


        out_a.backward(grad_a)
        self.data_owners[source_id-1].optimizer_a.step()

        return (loss, acc)

    def aggregate(self):
        total_samples = 0
        for i in range(len(self.data_owners)):
            print(i)
            total_samples += self.data_owners[i].num_of_samples
            if i == 0: # just copy
                k = 0
                for key in self.g_model_part_a.state_dict().keys():
                    self.g_model_part_a.state_dict()[key] = self.data_owners[i].model_part_a.state_dict()[key].clone()
                    self.g_model_part_a.state_dict()[key] = torch.multiply(self.g_model_part_a.state_dict()[key], self.data_owners[i].num_of_samples)
                    
                    if (i == len(self.data_owners) - 1):
                        self.g_model_part_a.state_dict()[key] = torch.divide(self.g_model_part_a.state_dict()[key], total_samples)

                for key in self.g_model_part_b.state_dict().keys():
                    self.g_model_part_b.state_dict()[key] = self.data_owners[i].model_part_b.state_dict()[key].clone()
                    self.g_model_part_b.state_dict()[key] = torch.multiply(self.g_model_part_b.state_dict()[key], self.data_owners[i].num_of_samples)

                    if (i == len(self.data_owners) - 1):
                        self.g_model_part_b.state_dict()[key] = torch.divide(self.g_model_part_b.state_dict()[key], total_samples)

                for key in self.g_model_part_c.state_dict().keys():
                    self.g_model_part_c.state_dict()[key] = self.data_owners[i].model_part_c.state_dict()[key].clone()
                    self.g_model_part_c.state_dict()[key] = torch.multiply(self.g_model_part_c.state_dict()[key], self.data_owners[i].num_of_samples)

                    if (i == len(self.data_owners) - 1):
                        self.g_model_part_c.state_dict()[key] = torch.divide(self.g_model_part_c.state_dict()[key], total_samples)
            else:
                for key in self.g_model_part_a.state_dict().keys():

                    self.g_model_part_a.state_dict()[key] += self.data_owners[i].model_part_a.state_dict()[key].clone()
                    self.g_model_part_a.state_dict()[key] = torch.multiply(self.g_model_part_a.state_dict()[key], self.data_owners[i].num_of_samples)

                    if (i == len(self.data_owners) - 1):
                        self.g_model_part_a.state_dict()[key] = torch.divide(self.g_model_part_a.state_dict()[key], total_samples)
                
                for key in self.g_model_part_b.state_dict().keys():
                    self.g_model_part_b.state_dict()[key] += self.data_owners[i].model_part_b.state_dict()[key].clone()
                    self.g_model_part_b.state_dict()[key] = torch.multiply(self.g_model_part_b.state_dict()[key], self.data_owners[i].num_of_samples)

                    if (i == len(self.data_owners) - 1):
                        self.g_model_part_b.state_dict()[key] = torch.divide(self.g_model_part_b.state_dict()[key], total_samples)

                for key in self.g_model_part_c.state_dict().keys():
                    self.g_model_part_c.state_dict()[key] += self.data_owners[i].model_part_c.state_dict()[key].clone()
                    self.g_model_part_c.state_dict()[key] = torch.multiply(self.g_model_part_c.state_dict()[key], self.data_owners[i].num_of_samples)

                    if (i == len(self.data_owners) - 1):
                        self.g_model_part_c.state_dict()[key] = torch.divide(self.g_model_part_c.state_dict()[key], total_samples)