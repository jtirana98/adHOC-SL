import adHocSL
import datasets.cifar_data as cifar_data

import torch
import numpy as np
import random

def main():
    sys_ = adHocSL.AdHocSL(pointa=3, pointb=5, num_dataowners=2, model_name="ignore for now")

    # Load Data NOTE: adapt accordingly
    SEED = 1234

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    TESTSIZE = 10000    
    GLOBALSIZE = 10000  # 10.000 samples (50,50,50,50,50,50,50,50,50,50)
    TRAINSIZE = 40000   # 20.000 samples per data owner (80,80,80,80,80,20,20,20,20,20) and (20,20,20,20,20,80,80,80,80,80)

    trainset, testset = cifar_data.get_dataset()
    trainloaderG, validloaderG, testloader = cifar_data.get_dataloaders(trainset, testset, batch_size=sys_.training_par.batch_size)
    trainloader1, validloader1, testloader = cifar_data.get_dataloaders(trainset, testset, batch_size=sys_.training_par.batch_size)
    trainloader2, validloader2, testloader = cifar_data.get_dataloaders(trainset, testset, batch_size=sys_.training_par.batch_size)
    
    train_iterator1 = iter(trainloaderG + trainloader1) # 10.000 samples (50,50,50,50,50,50,50,50,50,50) and 20.000 samples (80,80,80,80,80,20,20,20,20,20)
    valid_iterator1 = iter(validloaderG + validloader1) # 20% of train dataset
    test_iterator1 = iter(testloader)                   # 10.000 samples (50,50,50,50,50,50,50,50,50,50)    

    train_iterator2 = iter(trainloaderG + trainloader2) # 10.000 samples (50,50,50,50,50,50,50,50,50,50) and # 20.000 samples (20,20,20,20,20,80,80,80,80,80)
    valid_iterator2 = iter(validloaderG + validloader2) # 20% of train dataset
    test_iterator2 = iter(testloader)                   # 10.000 samples (50,50,50,50,50,50,50,50,50,50)
    
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    fedavg_step = 10
    warmup = 5 

    for epoch in range(sys_.training_par.epoch_num):
        train_iterator1 = iter(trainloader1)
        train_iterator2 = iter(trainloader2)
        #print(len(train_iterator))
        epoch_loss = 0
        epoch_acc = 0

        for batch1, batch2 in zip(train_iterator1, train_iterator2):
            images1, labels1 = batch1
            images2, labels2 = batch2

            (loss, acc) = sys_.local_update( 1, images1, labels1)
            print(f'{loss} {acc}')
            (loss, acc) = sys_.local_update( 2, images2, labels2)
            print(f'{loss} {acc}')

        if (epoch >= warmup and epoch % fedavg_step == 0):
            sys_.fed_avg()
            


if __name__ == '__main__':
    main()

