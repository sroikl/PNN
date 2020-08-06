import numpy as np
import glob
import pandas as pd
from tqdm import tqdm
import torch
from matplotlib import pyplot as plt
# from OldVersionCode.Configuration import pixelfulldict,dry_wet_cloth,start_date
import Configuration
from torchvision.transforms import Normalize,RandomHorizontalFlip,RandomRotation,ToTensor
from PIL import Image
import os
from datetime import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ImageLoader:

    def __init__(self,UseExp3_5,start_date,end_date):
        self.UseExp3_5= UseExp3_5
        self.device = device
        self.padsize = 300

        if UseExp3_5:
            self.LABEL_DIR= Configuration.exp3_labelloc
            self.DATA_DIR = Configuration.exp3_dataloc
            self.pixeldict = Configuration.exp3_pixelmap
            self.NormData= Configuration.exp3_norm_map
            strt_date_components = start_date.split('-')
            self.strt_date = datetime(int(strt_date_components[0]), int(strt_date_components[1]),
                                      int(strt_date_components[2]),
                                      int(strt_date_components[3]), int(strt_date_components[4]),
                                      int(strt_date_components[5]))

            end_date_components = end_date.split('-')
            self.end_date = datetime(int(end_date_components[0]), int(end_date_components[1]),
                                     int(end_date_components[2]),
                                     int(end_date_components[3]), int(end_date_components[4]),
                                     int(end_date_components[4]))

        else:
            self.LABEL_DIR = Configuration.exp1000_labelloc
            self.DATA_DIR = Configuration.exp1000_dataloc
            self.pixeldict = Configuration.exp1000_pixelmap
            self.NormData = Configuration.exp1000_norm_map

    def ImageCrop(self,date:str):

        ListOfCroppedImages = []
        file = glob.glob(self.DATA_DIR + date + '**/*.tiff')
        CatTensor= None
        if file and ('2020_03_15_01_20_01_LWIR' or '2020_03_15_01_10_01_LWIR' not in file):

            data= plt.imread(file[0])
            norm_dry= np.median(data[self.NormData['dry'][1][0]:self.NormData['dry'][1][1],self.NormData['dry'][0][0]:self.NormData['dry'][0][1]])
            norm_wet= np.median(data[self.NormData['wet'][1][0]:self.NormData['wet'][1][1],self.NormData['wet'][0][0]:self.NormData['wet'][0][1]])

            data= (data-norm_wet)/(norm_dry-norm_wet)
            data[data > 1] = 10
            data[data < 0] = 10
            # print(date)
            for key in self.pixeldict.keys():
                img = data[self.pixeldict[key][1][0]:self.pixeldict[key][1][1],self.pixeldict[key][0][0]:self.pixeldict[key][0][1]]

                xsize,ysize = img.shape
                padx = self.padsize - xsize ; pady = self.padsize - ysize
                Padded_im = np.pad(img,((padx//2,padx//2),(pady//2,pady//2)),constant_values=0,mode='constant')

                image_tensor = Image.fromarray(Padded_im)
                flipped= self.Random_flip(image_tensor)
                rotated = self.Random_Rotation(flipped)
                norm= self.Normalize_img(rotated)


                ListOfCroppedImages.append(norm.squeeze(dim=0))

            CatTensor = torch.stack([img for img in ListOfCroppedImages])

        return CatTensor

    @staticmethod
    def Random_Rotation(img):
        transform= RandomRotation(degrees=(0,0))
        rotated_img= transform.__call__(img)
        to_tensor= ToTensor()
        return to_tensor(rotated_img,)

    @staticmethod
    def Random_flip(img,p=0.1):
        transform = RandomHorizontalFlip(p=p)
        flipped_img= transform.__call__(img)
        return flipped_img

    @staticmethod
    def Normalize_img(img):
        # transform = Normalize(mean=(0,0,0),std=(1,1,1))
        # norm_img = transform.__call__(img)
        return img



class DataLoader(ImageLoader):
    def __init__(self,UseExp3_5:bool):
        super().__init__(UseExp3_5= UseExp3_5,start_date=Configuration.exp_args['start_date'],end_date=Configuration.exp_args['end_date'])
    def LoadData(self):

        if self.UseExp3_5:
            labeldict = self.Collect_exp3_dates_labels()
        else:
            labeldict = self.GetDateTimeLabel(self.LABEL_DIR,self.pixeldict)

        with tqdm(total=len(labeldict['lys1']),desc='Loading Data') as pbar:
            label_list,data_list = [],[]
            for i in range(len(labeldict['lys1'])):
                labels = [torch.Tensor(np.asarray(labeldict[key][i][0])) for key in labeldict.keys()] ; date = labeldict['lys1'][i][1]
                img_label= torch.stack([label for label in labels])
                img_tensor= self.ImageCrop(date=date)

                if img_tensor is not None:
                    data_list.append(img_tensor)
                    label_list.append(img_label)

                pbar.update(1)
            pbar.close()

            X= torch.stack([tensor for tensor in data_list],dim=0)
            y= torch.stack([label for label in label_list],dim=0)

        self.X = X ; self.y= y
        return X,y

    def Collect_exp3_dates_labels(self):

        labels_csv= pd.read_csv(self.LABEL_DIR)
        labeldict= create_labels_dict_exp3(self.pixeldict)

        time = np.asarray(labels_csv.iloc[:, 0])
        len_col = labels_csv.shape[0]

        with tqdm(total=len_col * 32, desc=f'exp3 collecting') as pbar:
            for col_num in range(1, len(labels_csv.columns)):
                labels = np.asarray(labels_csv.iloc[:, col_num].interpolate('linear').fillna(method='bfill'))
                for time_point, label in zip(time, labels):

                    #if working with hourly data
                    # sec_member = '_'
                    # thirt_member = time_point.split('T')[0].replace('-','_')
                    # first_member = time_point.split('T')[1][:-1].replace(':', '_')
                    # date_compare= datetime.fromisoformat(time_point[:-1])


                    #if working with data 'version1'
                    sec_member = '_'
                    thirt_member = time_point.split(' ')[0].replace('-','_')
                    first_member = time_point.split(' ')[1][:-1].replace(':', '_')
                    date_compare= datetime.strptime(time_point, '%Y-%m-%d %H:%M:%S')

                    if date_compare >= self.strt_date and date_compare <= self.end_date:
                        idx = labels_csv.columns.values[col_num].lower().find('_')
                        key= labels_csv.columns.values[col_num].lower()[idx + 1:].replace('_','')

                        try: #TODO: this is since not all lysemetrs are active
                            labeldict[key].append(
                                (label* 60. , ''.join((thirt_member, sec_member, first_member[:-1]))))
                        except KeyError:
                            continue

                    pbar.update()
        return labeldict

    @staticmethod
    def GetDateTimeLabel(LABEL_DIR:str,pixeldict:dict) -> dict:

        #this function takes the label input dir and based on the date&time returns a dictionary with keys lys1,lys2...and values
        #(label,date)

        # === initialize ===
        labeldict= create_labels_dict(pixeldict)
        labels_csv = pd.read_csv(LABEL_DIR)
        # === initialize dates and times of data ====
        dates = np.asarray('2019', dtype='datetime64[Y]') + np.asarray(labels_csv['day of year'],
                                                                       dtype='timedelta64[D]') - 1
        hours = np.asarray(labels_csv['hour'],dtype='int8') ; minutes = np.asarray(labels_csv['minute'],dtype='int8')

        # === Uploading the labels ===
        labels= labels_csv['ET'] ;labels= labels.interpolate(method='linear')
        plant_labels= np.asarray(labels_csv['lysimeter'])

        for i,label in enumerate(labels):
            labeldict[plant_labels[i]].append((label,str(dates[i]).replace('-','_') + '_{num:02d}_'.format(num=hours[i]) + '{num:02d}'.format(num=minutes[i])))
        return labeldict

    @staticmethod
    def GetDepthDates(DATA_DIR):
        list_dates_depth= []
        Depth_names= glob.glob(os.path.join(DATA_DIR,'**Depth_day_night*/'))
        for name in sorted(Depth_names):
            list_dates_depth.append(name.split('/')[-2][:19])

        return list_dates_depth



    @staticmethod
    def GetDateObject(date):
        date_object= datetime.now()
        try:
            date_object=datetime(int(date.split('_')[0]), int(date.split('_')[1]), int(date.split('_')[2]),
                    int(date.split('_')[3]), int(date.split('_')[4]))
        except ValueError:
            pass
        return date_object

    def N_points_MA(self, labels, n):
        filtered_labels = []
        labels_pad = np.pad(labels, (n - 1, 0), 'constant', constant_values=labels[0])
        for i in range(len(labels)):
            mean_point = np.mean(labels_pad[i:i + (n - 1)])
            filtered_labels.append(mean_point)

        return filtered_labels
def create_image_dict(pixeldict:dict) -> dict:
    imagedict = {}
    for key in pixeldict.keys():
        imagedict[key] = []
    return imagedict

def create_labels_dict(pixeldict:dict) -> dict:
    imagedict = {}
    for key in pixeldict.keys():
        imagedict['lys'+str(int(key))] = []
    return imagedict

def create_labels_dict_exp3(pixeldict:dict) -> dict:
    imagedict = {}
    for key in pixeldict.keys():
        key = key.replace('_', '')
        imagedict[key] = []
    return imagedict
