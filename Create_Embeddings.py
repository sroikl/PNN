from CollectData import CollectData
from Configuration import exp_args,line_dict,dataloc_dict,labelloc_dict,list_of_exp,list_of_keys
from PlantDataset import EmbeddingsDataset
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from Model import Encoder
import tqdm
import pickle

def Create_Embeddings():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Collect all the data from all available exp's
    DataObject= CollectData(dataloc_dict= dataloc_dict, labelloc_dict= labelloc_dict,list_of_exp= list_of_exp, list_of_keys= list_of_keys,
                            pad_size= 300,start_date= exp_args['start_date'],end_date=exp_args['end_date'])


    model= Encoder().to(device=device)
    Transforms= transforms.Compose([transforms.ToPILImage(),transforms.Resize((300,300)),transforms.ToTensor()])

    Embeddings_Dict= {}
    for exp in line_dict.keys():
        Embeddings_Dict[exp]= {}
        for line in line_dict[exp].keys():
            Embeddings_Dict[exp][line]={}
            for plant in line_dict[exp][line]:
                Embeddings_Dict[exp][line][plant]= []
                sampels= DataObject.ImageDict[exp][plant] ; labels= DataObject.LabelDict[exp][plant]
                wet_norms= DataObject.image_wet_norm[exp][plant] ; dry_norms= DataObject.image_dry_norm[exp][plant]
                dataset_= EmbeddingsDataset(file_list= sampels , label_list= labels, image_wet_norm= wet_norms,
                                       image_dry_norm= dry_norms, Transforms= Transforms)
                dl_= DataLoader(dataset= dataset_, batch_size=16, shuffle=False, drop_last=True)
                with tqdm.tqdm(total=len(dl_),desc=f'Embedding Plant {plant}') as pbar:
                    for x,y in dl_:
                        embeddings= model(x)
                        Embeddings_Dict[exp][line][plant].append((embeddings.squeeze(dim=1),y))
                        pbar.update()
                pbar.close()

    with open('Datasets.pkl','wb') as f:
        pickle.dump((Embeddings_Dict),f)

    return Embeddings_Dict
if __name__ =='__main__':
    Create_Embeddings()
