import torch
import numpy as np
import tqdm
import pandas as pd
import Configuration as cfg
import glob
from matplotlib import pyplot as plt
from datetime import datetime
import copy
class CollectData:
    def __init__(self,dataloc_dict:dict,labelloc_dict:dict,list_of_exp:list,list_of_keys:list,pad_size:int,start_date:str,end_date:str):

        self.padsize= pad_size
        strt_date_components= start_date.split('-')
        self.strt_date= datetime(int(strt_date_components[0]), int(strt_date_components[1]), int(strt_date_components[2]),
                         int(strt_date_components[3]), int(strt_date_components[4]),int(strt_date_components[5]))

        end_date_components= end_date.split('-')
        self.end_date= datetime(int(end_date_components[0]), int(end_date_components[1]), int(end_date_components[2]),
                         int(end_date_components[3]), int(end_date_components[4]),int(end_date_components[4]))

        self.dataloc_dict= dataloc_dict
        self.labelloc_dict= labelloc_dict
        self.ImageDict, self.LabelDict,self.DateDict,self.image_wet_norm,self.image_dry_norm= \
                                        init_exp_dicts(list_of_exp= list_of_exp,list_of_keys= list_of_keys)
        self.Collect_datatime_label()
        self.Collect_Images_from_Expiraments()

    def Collect_Images_from_Expiraments(self):
        #ThisFunction collects Images from an expirament
        for exp in self.dataloc_dict.keys():
            with tqdm.tqdm(total= self.get_len_(exp),desc=f'{exp} Data Collection') as pbar:
                for date in self.DateDict[exp][next(iter(self.DateDict[exp]))]:
                    self.Collect_Data_from_Image(date= date,exp= exp)
                    pbar.update()
            pbar.close()

    def get_len_(self,exp):
        for key in self.DateDict[exp].keys():
            len_= len(self.DateDict[exp][key])
            return len_

    def Collect_Data_from_Image(self,date,exp):
        potential_file= glob.glob(self.dataloc_dict[exp] + date + '**/*.tiff')

        if potential_file:
            image= plt.imread(potential_file[0])
            dry_tmp,wet_tmp= self.get_norm_vals(exp= exp,image=image)

            for key in self.ImageDict[exp].keys():
                img = image[cfg.pixelmap[exp][key][1][0]:cfg.pixelmap[exp][key][1][1],
                                cfg.pixelmap[exp][key][0][0]:cfg.pixelmap[exp][key][0][1]]

                self.ImageDict[exp][key].append(img)
                self.image_wet_norm[exp][key].append(wet_tmp)
                self.image_dry_norm[exp][key].append(dry_tmp)

    def get_norm_vals(self,image,exp):

        dry_tmp= np.median(image[cfg.norm_map[exp]['dry'][1][0]:cfg.norm_map[exp]['dry'][1][1],
                            cfg.norm_map[exp]['dry'][0][0]:cfg.norm_map[exp]['dry'][0][1]])

        wet_tmp= np.median(image[cfg.norm_map[exp]['wet'][1][0]:cfg.norm_map[exp]['wet'][1][1],
                            cfg.norm_map[exp]['wet'][0][0]:cfg.norm_map[exp]['wet'][0][1]])


        return dry_tmp,wet_tmp

    def Collect_datatime_label(self):

        #this function takes the label input dir and based on the date&time returns a dictionary with keys lys1,lys2...and values
        #(label,date)

        for key in self.labelloc_dict.keys():
            labels_csv = pd.read_csv(self.labelloc_dict[key])
            if key == 'exp1000':
                self.Collect_exp1000_dates_labels(labels_csv= labels_csv, exp= key)
            elif key== 'exp3':
                self.Collect_exp3_dates_labels(labels_csv=labels_csv, exp= key)

    def Collect_exp1000_dates_labels(self,labels_csv,exp):

        # === initialize dates and times of data ====
        dates = np.asarray('2019', dtype='datetime64[Y]') + np.asarray(labels_csv['day of year'],
                                                                       dtype='timedelta64[D]') - 1
        hours = np.asarray(labels_csv['hour'], dtype='int8');
        minutes = np.asarray(labels_csv['minute'], dtype='int8')

        # === Uploading the labels ===
        labels = np.asarray(labels_csv['ET'].interpolate(method='quadratic').fillna(0))
        labels -= labels.min()
        labels /= max(labels)
        labels *= 100
        plant_labels = labels_csv['lysimeter']
        len_labels= len(labels)
        with tqdm.tqdm(total= len_labels, desc=f'{exp} label collecting') as pbar:
            for min,hr,date,label,idx in zip(minutes,hours,dates,labels,range(len(plant_labels))):
                self.LabelDict[exp][plant_labels[idx]].append(label)
                self.DateDict[exp][plant_labels[idx]].append(str(date).replace('-', '_') + '_{num:02d}_'.format(
                    num=hr) + '{num:02d}'.format(num=min))
                pbar.update()

    def Collect_exp3_dates_labels(self,labels_csv,exp):
        time= np.asarray(labels_csv.iloc[:,0])
        len_col= labels_csv.shape[0]

        # === for exp3 we have specifically measured temps of cloths
        self.wet_cloth_temp= np.asarray(labels_csv.iloc[:,-1])
        self.dry_cloth_temp= np.asarray(labels_csv.iloc[:,-2])

        with tqdm.tqdm(total= len_col*32, desc=f'{exp} collecting') as pbar:
            for col_num in range(1,len(labels_csv.columns)-2):
                labels= np.asarray(labels_csv.iloc[:,col_num].interpolate(method='quadratic').fillna(0.))
                # labels= self.N_points_MA(labels,10)
                # labels-= labels.min()
                # labels/= max(labels)
                # labels *= 100

                for time_point,label in zip(time,labels):
                    sec_member= '_'
                    thirt_member= time_point.split()[1].replace(':','_')
                    first_member= time_point.split()[0].replace('-','_')
                    date_compare=datetime(int(first_member.split('_')[0]),int(first_member.split('_')[1]),int(first_member.split('_')[2]),
                                          int(thirt_member.split('_')[0]),int(thirt_member.split('_')[1]),int(thirt_member.split('_')[2]))
                    if date_compare >= self.strt_date and date_compare <= self.end_date:
                        self.LabelDict[exp][labels_csv.columns.values[col_num].lower()].append(label)
                        self.DateDict[exp][labels_csv.columns.values[col_num].lower()].append(''.join((first_member,sec_member,thirt_member)))
                    pbar.update()
                self.N_points_MA(self.LabelDict[exp][labels_csv.columns.values[col_num].lower()],3)

    def N_points_MA(self,labels,n):
        filtered_labels= []
        labels_pad= np.pad(labels,(n-1,0),'constant', constant_values=labels[0])
        for i in range(len(labels)):
            mean_point= np.mean(labels_pad[i:i+(n-1)])
            filtered_labels.append(mean_point)


        filtered_labels= self.Fit2BW(filtered_labels)
        # labels= self.Fit2BW(labels)

        return np.asarray(filtered_labels)

    def Fit2BW(self,series_):
        series= np.asarray(copy.deepcopy(series_))
        series -= min(series)
        series /= max(series)
        series *= 100
        return series
def init_exp_dicts(list_of_exp:list,list_of_keys: list):
    '''

    :param list_of_exp: list of names of exp to be used
    :param list_of_keys: list of lists( as the number of expiraments) - each with the names of the plants,
                        format= [[list of exp1 plants],[list of exp2 plants],...]
    :return:
    '''
    datedict,imagedict,labeldict,dry_norm_dict,wet_norm_dict= {},{},{},{},{}
    for i,exp in enumerate(list_of_exp):
        labeldict[exp],imagedict[exp],datedict[exp],dry_norm_dict[exp],wet_norm_dict[exp]= {},{},{},{},{}
        for key in list_of_keys[i]:
            labeldict[exp][key], imagedict[exp][key],datedict[exp][key],dry_norm_dict[exp][key],wet_norm_dict[exp][key]= [],[],[],[],[]

    return imagedict,labeldict,datedict,dry_norm_dict,wet_norm_dict

if __name__=='__main__':

    exp1000_dataloc = "//Users/roiklein/Dropbox/Msc Project/Deep Learning Project/Exp1000_Full/"
    exp1000_labelloc = '/Users/roiklein/Dropbox/Msc Project/Deep Learning Project/lys_prc.csv'

    exp3_dataloc = '/Users/roiklein/Desktop/PNN/DL_TC_EXP_3-5.csv'
    exp3_labelloc = '/Users/roiklein/Desktop/PNN/DL_TC_EXP_3-5.csv'

    dataloc_dict = dict(exp1000=exp1000_dataloc, exp3=exp3_dataloc)
    labelloc_dict = dict(exp1000=exp1000_labelloc, exp3=exp3_dataloc)


    data = CollectData(dataloc_dict= dataloc_dict, labelloc_dict= labelloc_dict,
                       list_of_exp= list_of_exp,list_of_keys= list_of_keys)