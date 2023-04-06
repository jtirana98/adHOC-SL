
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
    def __init__(self, id, model_a, model_b, model_c, training_par):
        self.id = id
        self.model_a = model_a
        self.model_b = model_b
        self.model_c = model_c
        self.training_par = training_par

        # define a different optimizer for each model part
        self.optimizer_a = Adam(self.model_a.parameters(), lr=training_par.lr)  # NOTE: adapt accordingly
        self.optimizer_b = Adam(self.model_b.parameters(), lr=training_par.lr)
        self.optimizer_c = Adam(self.model_c.parameters(), lr=training_par.lr)

        self.criterion = nn.CrossEntropyLoss()                                  # NOTE: adapt accordingly


class AdHocSL:
    data_owners = []

    def __init__(self, pointa, pointb, num_dataowners, model_name):  # TODO: pass the model definition
        self.pointa = pointa
        self.pointb = pointb
        self.training_par = define_training_param()

        g_model_a = my_resnet.get_resnet18(self.training_par.num_out, -1, self.pointa)
        g_model_b = my_resnet.get_resnet18(self.training_par.num_out, self.pointa, self.pointb)
        g_model_c = my_resnet.get_resnet18(self.training_par.num_out, self.pointb, -1)
        

        for i in range(num_dataowners):
            model_a = my_resnet.get_resnet18(self.training_par.num_out, -1, self.pointa)
            model_b = my_resnet.get_resnet18(self.training_par.num_out, self.pointa, self.pointb)
            model_c = my_resnet.get_resnet18(self.training_par.num_out, self.pointb, -1)
            
            model_a.load_state_dict(copy.deepcopy(g_model_a.state_dict()))
            model_b.load_state_dict(copy.deepcopy(g_model_b.state_dict()))
            model_c.load_state_dict(copy.deepcopy(g_model_c.state_dict()))
            
            self.data_owners.append(DataOwner(i, model_a, model_b, model_c, self.training_par))




    def local_update(self, d_id, input, label):
        self.data_owners[d_id-1].optimizer_a.zero_grad()
        self.data_owners[d_id-1].optimizer_b.zero_grad()
        self.data_owners[d_id-1].optimizer_c.zero_grad()

        # start forward propagation

        out_a = self.data_owners[d_id-1].model_a(input)
        det_out_a = out_a.clone().detach().requires_grad_(True)


        out_b = self.data_owners[d_id-1].model_b(det_out_a)
        det_out_b = out_b.clone().detach().requires_grad_(True)

        out_c = self.data_owners[d_id-1].model_c(det_out_b)
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

        out_a = self.data_owners[source_id-1].model_a(input)
        det_out_a = out_a.clone().detach().requires_grad_(True)


        out_b = self.data_owners[destination_id-1].model_b(det_out_a)
        det_out_b = out_b.clone().detach().requires_grad_(True)

        out_c = self.data_owners[source_id-1].model_c(det_out_b)
        
        loss = self.data_owners[source_id-1].criterion(out_c, label)  # calculates cross-entropy loss
        
        # start backward propagation

        loss.backward() 

        loss = loss.item()
        acc = cifar_data.categorical_accuracy(out_c, label)
        acc = acc.item()

        self.data_owners[source_id-1].optimizer_c.step()

        grad_b = det_out_b.grad.clone().detach()
        out_b.backward(grad_b)
        self.data_owners[destination_id-1].optimizer_b.step()

        grad_a = det_out_a.grad.clone().detach()
        out_a.backward(grad_a)
        self.data_owners[source_id-1].optimizer_a.step()

        return (loss, acc)
