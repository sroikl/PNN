import torch
from Configuration import exp_args
from DataLoader import DataLoader
from Model import TemporalSpatialModel
from TrainingOldVersion import TCNTrainer
import numpy as np
from torch.utils.data.sampler import BatchSampler,SequentialSampler
import os
from random import choices
import itertools
from torch.utils.data import Dataset
def runTCN():


    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data = DataLoader(UseExp3_5=True)
    X, y_tr= data.LoadData()

    #Create Plants labels
    num_samples,num_plants= y_tr.shape
    y_pl= torch.arange(num_plants)
    y_pl= y_pl.repeat(num_samples,1)

    validation_split = 0.2
    dataset = torch.utils.data.TensorDataset(X, y_tr,y_pl)
    dl_train, dl_test = Split_Test_Train(dataset=dataset, validation_split=validation_split,
                                         batch_size=exp_args['batch_size'],shuffle_test=True)

    print(f'Sampels Shape is:%s' % {next(iter(dl_train))[0].shape})
    print(f'Transpiration Label Shape is:%s' % {next(iter(dl_train))[1].shape})
    print(f'Plant Label Shape is:%s' % {next(iter(dl_train))[2].shape})

    model = TemporalSpatialModel(num_levels=exp_args['tcn_num_levels'], num_hidden=exp_args['tcn_hidden_channels'],
                      embedding_size=exp_args['embedding_dim'], kernel_size=exp_args['tcn_kernel_size'],
                      dropout=exp_args['tcn_dropout'],num_plants= num_plants).to(device=device)

    optimizer = torch.optim.Adam(
            model.parameters(), betas=(0.9, 0.999), lr=exp_args['lr'])

    # if os.path.exists(f'{os.getcwd()}/Model.pt'):
    #     model.load_state_dict(torch.load(f'{os.getcwd()}/Model.pt'))

    Transpiration_loss_fn= torch.nn.MSELoss()
    Plant_loss_fn= torch.nn.CrossEntropyLoss().to()
    datadict= dict(dl_train= dl_train, dl_test= dl_test)
    torch.save(datadict, f'{os.getcwd()}/Data.pt')
    Trainer= TCNTrainer(model=model,Transpiration_loss_fn= Transpiration_loss_fn,Plant_loss_fn=Plant_loss_fn,optimizer=optimizer,device=device,num_plants=num_plants)
    Trainer.fit(dl_train=dl_train,dl_test=dl_test,num_epochs=exp_args['num_epochs'])

def Split_Test_Train(dataset,validation_split,batch_size,shuffle_test=True):

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    num_full_batches= dataset_size//batch_size

    indices= np.arange(num_full_batches*batch_size)
    indices= np.reshape(indices,(num_full_batches,batch_size))

    num_full_batches_test= int(num_full_batches*validation_split)
    batchs= np.arange(0,num_full_batches)
    if shuffle_test:
        batch_indices_test= choices(batchs,k=num_full_batches_test)
    else:
        batch_indices_test= batchs[-num_full_batches_test:]
    batch_indices_train= [ val for val in np.arange(0,num_full_batches) if val not in batch_indices_test]

    train_indices= indices[batch_indices_train]
    test_indices= indices[batch_indices_test]

    # Creating PT data samplers and loaders:
    train_sampler = BatchSampler(SequentialSampler(list(itertools.chain(*train_indices))),batch_size=batch_size,drop_last=True)
    valid_sampler = BatchSampler(SequentialSampler(list(itertools.chain(*test_indices))),batch_size=batch_size,drop_last=True)

    dl_train = torch.utils.data.DataLoader(dataset,
                                           sampler=train_sampler, shuffle=False)
    dl_test = torch.utils.data.DataLoader(dataset,
                                          sampler=valid_sampler, shuffle=False)

    return dl_train,dl_test

# class PlantsDataset(Dataset):
#     def __init__(self,PlantsData,TranspirationLabels,PlantsLabels):
#         self.sampels= []
#         for

if __name__ == '__main__':
    runTCN()