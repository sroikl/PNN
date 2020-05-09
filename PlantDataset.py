from torch.utils.data import Dataset
import torch
import numpy as np
from torch import nn
import torchvision
import tqdm
import matplotlib.pyplot as plt

class EmbeddingsDataset(Dataset,nn.Module):
    def __init__(self,file_list,label_list,image_wet_norm,image_dry_norm,Transforms= None):
        super(EmbeddingsDataset,self).__init__()
        self.sampels= []

        for sample,label,wet_norm,dry_norm in zip(file_list,label_list,image_wet_norm,image_dry_norm):
            #Apply transforms on samples
            if Transforms:
                sample= Transforms(np.uint8(sample))
            sample= self.norm_im(data= sample, wet= wet_norm, dry= dry_norm)
            self.sampels.append((sample.repeat(3,1,1).unsqueeze(dim=0),label))


    def __len__(self):
        return len(self.sampels)

    def __getitem__(self, idx):
        return self.sampels[idx]


    def norm_im(self,data, dry, wet):

        data = (data - wet) / (dry - wet)
        data[data > 1] = 10
        data[data < 0] = 10

        return data

class PlantDataset(Dataset):
    def __init__(self,samples,labels,Transforms):
        super(PlantDataset,self).__init__()

        self.samples= []

        for sample,label in zip(samples,labels):

            if Transforms:
                sample= Transforms(sample)

            self.samples.append((sample,label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

