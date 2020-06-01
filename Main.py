from Training import train_model,eval_model
from Model import TCN_Model
from Configuration import exp_args
from PlantDataset import PlantDataset
import os
import torch
from torch.utils.data import random_split,DataLoader
from Create_Embeddings import Create_Embeddings
import pickle
import itertools
import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler
from TrainingOldVersion import TCNTrainer
def RunExpirement(train_lines:list,test_lines:list):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if os.path.exists(f'{os.getcwd()}/Datasets.pkl'):
        print('====== Loading Embeddings ======\n')
        with open(f'{os.getcwd()}/Datasets.pkl','rb') as f:
            Embeddings_Dict= pickle.load(f)
    else:
        Embeddings_Dict= Create_Embeddings()


    train_sampels,train_labels= [],[]
    test_sampels, test_labels= [],[]

    print('====== Unpacking Embeddings ======\n')
    for exp in Embeddings_Dict.keys():
        for line in Embeddings_Dict[exp].keys():
            for plant in Embeddings_Dict[exp][line].keys():
                samples,labels= zip(*Embeddings_Dict[exp][line][plant]) # list of tuples
                if line in train_lines:
                    train_sampels.append(samples) ; train_labels.append(labels)
                elif line in test_lines:
                    test_sampels.append(samples) ; test_labels.append(labels)
    train_sampels= list(itertools.chain(*train_sampels))
    train_labels= list(itertools.chain(*train_labels))

    test_samples= list(itertools.chain(*test_sampels))
    test_labels= list(itertools.chain(*test_labels))
    # with tqdm.tqdm(total= len(train_sampels), desc='Fitting Standart Scaler') as pbar:
    #     data= np.concatenate((train_sampels[0].numpy(),train_sampels[1].numpy()))
    #     for i in range(1,len(train_sampels)):
    #         data= np.concatenate((data,train_sampels[i]))
    #         pbar.update()
    # scaler= StandardScaler()
    # scaler.fit(data)
    # print(f'Size of Standart Scaler mean vector:{len(scaler.mean_)}')
    # print(f'Size of Standart Scaler Var vector:{len(scaler.var_)}')

    scaler= None
    dataset_train= PlantDataset(samples= train_sampels , labels= train_labels,Transforms= None,scaler=scaler)
    dataset_test= PlantDataset(samples= test_samples, labels= test_labels, Transforms= None,scaler=scaler)

    # ForTest= True
    ForTest= False

    if ForTest:
        lenDataset = len(dataset_train)
        lenTrainset = int(lenDataset * 0.9)
        lenValset = lenDataset- lenTrainset
        train_set, val_set = random_split(dataset_train, [lenTrainset, lenValset])

        dataloaders = {'train': DataLoader(train_set,batch_size= exp_args['batch_size'],shuffle= True,drop_last=True),
                       'val': DataLoader(val_set,batch_size= exp_args['batch_size'],shuffle= True,drop_last=True),
                       'test': DataLoader(dataset_test,batch_size= exp_args['batch_size'],shuffle= False,drop_last=True)}
    else:
        lenDataset = len(dataset_train)
        lenTrainset = int(lenDataset * 0.7)
        lenTestset= int(lenDataset * 0.2)
        lenValset = lenDataset - lenTrainset- lenTestset
        train_set, val_set,test_set = random_split(dataset_train, [lenTrainset, lenValset,lenTestset])

        dataloaders = {'train': DataLoader(train_set, batch_size=exp_args['batch_size'], shuffle=True, drop_last=True),
                       'val': DataLoader(val_set, batch_size=exp_args['batch_size'], shuffle=True, drop_last=True),
                       'test': DataLoader(test_set, batch_size=exp_args['batch_size'], shuffle=False,
                                          drop_last=True)}

    X,y= next(iter(dataloaders['train']))
    print(f'Shape of Input Tensor: {X.shape}\n')
    print(f'Shape of Label Tensor: {y.shape}\n')

    print('====== Building Model ======\n')
    model= TCN_Model(num_levels= exp_args['tcn_num_levels'], num_hidden= exp_args['tcn_hidden_channels'],
                     embedding_size= exp_args['embedding_dim'],kernel_size=exp_args['tcn_kernel_size'],
                     dropout= exp_args['tcn_dropout']).double().to(device=device)

    if torch.cuda.device_count() >= 0:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    # params_non_frozen = filter(lambda p: p.requires_grad, model.parameters())
    optimizer= torch.optim.Adam(params= model.parameters() ,lr= exp_args['lr'])
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    criterion= torch.nn.MSELoss(reduction='mean').to(device=device)

    UseNewTraining= True
    # UseNewTraining= False

    if UseNewTraining:
        best_model,train_loss_list,val_lost_list= train_model(model=model, optimizer=optimizer,dataloaders= dataloaders,scheduler=lr_sched,device=device,
                            criterion=criterion, num_epochs= exp_args['num_epochs'])

        test_loss,true_labels,predicted_output= eval_model(model= best_model, dataloaders= dataloaders, criterion= criterion, optimizer= optimizer,
                          scheduler= None, device= device, num_epochs= 1)

        exp_data= dict(train_loss= train_loss_list, val_loss= val_lost_list,test_loss= test_loss,true_labels=true_labels,predicted_output=predicted_output)
        torch.save(exp_data,f'{os.getcwd()}/Results.pt')

    else:

        Trainer = TCNTrainer(model=model, loss_fn=criterion, optimizer=optimizer, device=device)
        Trainer.fit(dl_train=dataloaders['train'], dl_test=dataloaders['test'], num_epochs=exp_args['num_epochs'])

if __name__ =='__main__':
    RunExpirement(train_lines=['line1','line2','line3','line4',],test_lines=['line5'])
    # RunExpirement(train_lines=['line1'], test_lines=['line2'])

