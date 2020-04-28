from torch.utils.data import Dataset
import torch
import numpy as np
from torch import nn
import torchvision
import tqdm
import matplotlib.pyplot as plt

class PlantDataset(Dataset,nn.Module):
    def __init__(self,file_list,label_list,image_wet_norm,image_dry_norm,Transforms= None,desc= 'Train'):
        super(PlantDataset,self).__init__()
        self.sampels= []
        len_= len(file_list[0])
        iter_sampels= iter(zip(*file_list)) ; iter_labels= iter(zip(*label_list))
        iter_wet= iter(zip(*image_wet_norm)) ; iter_dry= iter(zip(*image_dry_norm))

        with tqdm.tqdm(total= len_, desc= desc) as pbar:
            for i in range(len_):
                samples= next(iter_sampels) ; next_labels= next(iter_labels)
                next_wet_norm= next(iter_wet) ; next_dry_norm= next( iter_dry)
                meta_sample,meta_label= [],[]
                for sample,label,wet_norm,dry_norm in zip(list(samples),list(next_labels),list(next_wet_norm),list(next_dry_norm)):
                    #Apply transforms on samples
                    if Transforms:
                        sample= Transforms(np.uint8(sample))

                    sample= self.norm_im(data= sample, wet= wet_norm, dry= dry_norm)
                    to_tensor= torchvision.transforms.ToTensor()
                    sample= to_tensor(sample).repeat(3,1,1)
                    label= torch.Tensor(np.asarray(label))
                    meta_sample.append(sample)
                    meta_label.append(label)

                sample= torch.stack([img for img in meta_sample])
                label= torch.stack([lbl for lbl in meta_label])
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
