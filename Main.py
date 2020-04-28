from Training import train_model,eval_model
from Model import TCN_Model
from CollectData import CollectData
from Configuration import exp_args,line_dict,dataloc_dict,labelloc_dict,list_of_exp,list_of_keys
from PlantDataset import PlantDataset
from torchvision import transforms
import os
import torch
from torch.utils.data import ConcatDataset,random_split,DataLoader
import numpy as np

def RunExpirement(train_lines:list,test_lines:list):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Collect all the data from all available exp's
    DataObject= CollectData(dataloc_dict= dataloc_dict, labelloc_dict= labelloc_dict,
                            list_of_exp= list_of_exp, list_of_keys= list_of_keys)


    Transforms= transforms.Compose([transforms.ToPILImage(),transforms.Resize((300,300))])
    #Parse the data per lines
    train_sampels,train_labels,train_wet_norms,train_dry_norms= [],[],[],[]
    test_sampels, test_labels, test_wet_norms, test_dry_norms = [], [], [], []
    for exp in line_dict.keys():
        for line in line_dict[exp].keys():
            for plant in line_dict[exp][line]:
                sampels= DataObject.ImageDict[exp][plant] ; labels= DataObject.LabelDict[exp][plant]
                wet_norms= DataObject.image_wet_norm[exp][plant] ; dry_norms= DataObject.image_dry_norm[exp][plant]
                if line in train_lines:
                    train_sampels.append(sampels)
                    train_labels.append(labels)
                    train_wet_norms.append(wet_norms)
                    train_dry_norms.append(dry_norms)
                elif line in test_lines:
                    test_sampels.append(sampels)
                    test_labels.append(labels)
                    test_wet_norms.append(wet_norms)
                    test_dry_norms.append(dry_norms)

    dataset_train= PlantDataset(file_list= train_sampels , label_list= train_labels, image_wet_norm= train_wet_norms,
                                image_dry_norm= train_dry_norms, Transforms= Transforms,desc= 'Build Train Dataset')
    dataset_test= PlantDataset(file_list= test_sampels, label_list= test_labels, image_wet_norm= test_wet_norms,
                               image_dry_norm= test_dry_norms, Transforms= Transforms,desc= 'Build Test Dataset')

    lenDataset = len(dataset_train)
    lenTrainset = int(lenDataset * 0.9)
    lenValset = lenDataset- lenTrainset
    train_set, val_set = random_split(dataset_train, [lenTrainset, lenValset])

    dataloaders = {'train': DataLoader(train_set,batch_size= exp_args['batch_size'],shuffle= False,drop_last=True),
                   'val': DataLoader(val_set,batch_size= exp_args['batch_size'],shuffle= False,drop_last=True),
                   'test': DataLoader(dataset_test,batch_size= exp_args['batch_size'],shuffle= False,drop_last=True)}

    model= TCN_Model(num_levels= exp_args['tcn_num_levels'], num_hidden= exp_args['tcn_hidden_channels'],
                     embedding_size= exp_args['embedding_dim'],kernel_size=exp_args['tcn_kernel_size'],
                     dropout= exp_args['tcn_dropout'],encoder_name='Inception').double().to(device=device)

    if torch.cuda.is_available():
        model= torch.nn.DataParallel(model).to(device=device)

    optimizer= torch.optim.Adam(params= model.parameters(),lr= exp_args['lr'])
    criterion= torch.nn.MSELoss(reduction='mean').to(device=device)

    best_model,train_loss_list,val_lost_list= train_model(model=model, optimizer=optimizer,dataloaders= dataloaders,scheduler=None,device=device,
                            criterion=criterion, num_epochs= exp_args['num_epochs'])

    test_loss= eval_model(model= best_model, dataloaders= dataloaders, criterion= criterion, optimizer= optimizer,
                          scheduler= None, device= device, num_epochs= 1)

    exp_data= dict(model= best_model.state_dict(),train_loss= train_loss_list, val_loss= val_lost_list,test_loss= test_loss)
    torch.save(exp_data,f'{os.getcwd()}/ExpData.pt')

if __name__ =='__main__':
    # RunExpirement(train_lines=['line1','line2','line3','line4','line5'],test_lines=['line6','line7'])
    RunExpirement(train_lines=['line7',],test_lines=['line1','line2','line3','line4','line5','line6'])
