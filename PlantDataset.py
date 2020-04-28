from torch.utils.data import Dataset
import numpy as np
from torch import nn
import torchvision
import tqdm
import matplotlib.pyplot as plt

class PlantDataset(Dataset,nn.Module):
    def __init__(self,file_list,label_list,image_wet_norm,image_dry_norm,Transforms= None,desc= 'Train'):
        super(PlantDataset,self).__init__()
        self.sampels= []

        with tqdm.tqdm(total= len(file_list), desc= desc) as pbar:
            for sample,label,dry_norm,wet_norm in zip(file_list,label_list,image_dry_norm,image_wet_norm):

                #Apply transforms on samples
                if Transforms:
                    sample= Transforms(np.uint8(sample))

                sample= self.norm_im(data= sample, wet= wet_norm, dry= dry_norm)
                to_tensor= torchvision.transforms.ToTensor()
                sample= to_tensor(sample).repeat(3,1,1)
                self.sampels.append((sample,label))
                pbar.update()

    def __len__(self):
        return len(self.sampels)

    def __getitem__(self, idx):
        return self.sampels[idx]


    def norm_im(self,data, dry, wet):

        data = (data - wet) / (dry - wet)
        data[data > 1] = 10
        data[data < 0] = 10

        return data
