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

    trainset, testset = cifar_data.get_dataset()
    trainloader, validloader, testloader = cifar_data.get_dataloaders(trainset, testset, batch_size=sys_.training_par.batch_size)
    
    train_iterator = iter(trainloader)
    valid_iterator = iter(validloader)
    test_iterator = iter(testloader)
    
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    sl_step=5
    fedavg_step=10
    warmup=5
    datasharingM=0.5
    share_batch = 1 / datasharingM -1 
    

    for epoch in range(sys_.training_par.epoch_num):
        train_iterator1 = iter(trainloader)
        train_iterator2 = iter(trainloader)
        #print(len(train_iterator))
        epoch_loss = 0
        epoch_acc = 0

        for batch1, batch2 in train_iterator:
            images1, labels1 = batch1
            images2, labels2 = batch2

            if (epoch >= warmup and epoch % sl_step !=0):
                (loss, acc) = sys_.local_update( 1, images1, labels1)
                print(f'{loss} {acc}')
                (loss, acc) = sys_.local_update( 2, images2, labels2)
                print(f'{loss} {acc}')
            else:
                (loss, acc) = sys_.adHoc_update( 1, 2, images1, labels1)
                print(f'{loss} {acc}')
                (loss, acc) = sys_.adHoc_update( 2, 1, images2, labels2)
                print(f'{loss} {acc}')
        
        if (epoch >= warmup and epoch % fedavg_step == 0):
            sys_.fed_avg()


if __name__ == '__main__':
    main()

