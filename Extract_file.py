import numpy as np
import torch
import os
def Extract_file(load_path,name):
    train_loss_list,val_loss_list,test_loss_list=[],[],[]
    data = torch.load(load_path)
    train_loss= data['train_loss'] ; val_loss= data['val_loss'] ; test_loss= data['test_loss']
    for epoch_loss in train_loss:
        train_loss_list.append(epoch_loss.numpy())
    for epoch_loss in val_loss:
        val_loss_list.append(epoch_loss.numpy())
    for epoch_loss in test_loss:
        test_loss_list.append(epoch_loss.numpy())

    exp_data = dict( train_loss=train_loss_list, val_loss=val_loss_list,
                    test_loss=test_loss)
    torch.save(exp_data,f'{os.getcwd()}/{name}')
if __name__ =='__main__':
    name='Results.pt'
    load_path= f'{os.getcwd()}/{name}'
    Extract_file(load_path=load_path,name=name)