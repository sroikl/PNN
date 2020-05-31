import numpy as np
import torch
import os
def Extract_file(load_path,name):
    list_=[]
    data = torch.load(load_path)
    for epoch_loss in data:
        list_.append(epoch_loss.numpy())

    torch.save(f'{os.getcwd()}/{name}')
if __name__ =='__main__':
    name='Results.pt'
    load_path= f'{os.getcwd()}/{name}'
    Extract_file(load_path=load_path,name=name)