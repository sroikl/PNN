from Training import train_model,eval_model
from Model import TCN_Model
from CollectData import CollectData
from Configuration import exp_args,line_dict,dataloc_dict,labelloc_dict,list_of_exp,list_of_keys
from PlantDataset import PlantDataset
from torchvision import transforms
import os
import torch
from torch.utils.data import random_split,DataLoader
from Create_Embeddings import Create_Embeddings
import pickle
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

    train_sampels = lambda train_sampels: [item for sublist in train_sampels for item in sublist]
    train_labels= lambda train_labels: [item for sublist in train_labels for item in sublist]

    test_samples= lambda test_samples: [item for sublist in test_samples for item in sublist]
    test_labels= lambda test_labels: [item for sublist in test_labels for item in sublist]

    dataset_train= PlantDataset(samples= train_sampels , labels= train_labels,Transforms= None)
    dataset_test= PlantDataset(samples= test_samples, labels= test_labels, Transforms= None)

    lenDataset = len(dataset_train)
    lenTrainset = int(lenDataset * 0.9)
    lenValset = lenDataset- lenTrainset
    train_set, val_set = random_split(dataset_train, [lenTrainset, lenValset])

    dataloaders = {'train': DataLoader(train_set,batch_size= exp_args['batch_size'],shuffle= True,drop_last=True),
                   'val': DataLoader(val_set,batch_size= exp_args['batch_size'],shuffle= True,drop_last=True),
                   'test': DataLoader(dataset_test,batch_size= exp_args['batch_size'],shuffle= True,drop_last=True)}

    print('====== Building Model ======')
    model= TCN_Model(num_levels= exp_args['tcn_num_levels'], num_hidden= exp_args['tcn_hidden_channels'],
                     embedding_size= exp_args['embedding_dim'],kernel_size=exp_args['tcn_kernel_size'],
                     dropout= exp_args['tcn_dropout'],encoder_name='Inception').double().to(device=device)

    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    params_non_frozen = filter(lambda p: p.requires_grad, model.parameters())
    optimizer= torch.optim.Adam(params= params_non_frozen ,lr= exp_args['lr'])
    lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.05, patience=5, )

    criterion= torch.nn.MSELoss(reduction='mean').to(device=device)

    best_model,train_loss_list,val_lost_list= train_model(model=model, optimizer=optimizer,dataloaders= dataloaders,scheduler=lr_sched,device=device,
                            criterion=criterion, num_epochs= exp_args['num_epochs'])

    test_loss= eval_model(model= best_model, dataloaders= dataloaders, criterion= criterion, optimizer= optimizer,
                          scheduler= None, device= device, num_epochs= 1)

    exp_data= dict(model= best_model.state_dict(),train_loss= train_loss_list, val_loss= val_lost_list,test_loss= test_loss)
    torch.save(exp_data,f'{os.getcwd()}/ExpData.pt')

if __name__ =='__main__':
    RunExpirement(train_lines=['line1','line2','line3','line4','line5'],test_lines=['line6'])
    # RunExpirement(train_lines=['line7',],test_lines=['line1','line2','line3','line4','line5','line6'])
