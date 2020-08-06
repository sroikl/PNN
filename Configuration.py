from os.path import expanduser
import os
from datetime import datetime

CD= os.getcwd()
now= datetime.now()
SAVE_DIR = os.path.join(CD, 'Exp %s_%s_%s_%s_%s_%s' % (now.strftime("%m"), now.strftime("%d"), now.strftime("%Y"),
                                                       now.strftime("%H"), now.strftime("%M"), now.strftime("%S")))
# === PixelMaps ===
exp1000_pixelmap = {'lys1':((490,584),(84,200)),'lys2':((510,638),(256,378)),'lys3':((395,499),(46,198)),'lys4':((375,493),(246,384)),
             'lys5':((269,373),(86,194)),'lys6':((252,372),(232,380)),'lys7':((157,267),(39,191)),'lys8':((137,247),(235,375)),
             'lys9':((38,150),(117,221)),'lys10':((10,124),(258,386))}

# exp3_pixelmap= {'lys_1':((508,578),(26,76)),'lys_2':((512,582),(75,135)),'lys_3':((500,570),(129,189)),'lys_4':((508,578),(184,244)),
#                 'lys_5':((516,586),(264,324)),'lys_6':((516,586),(318,378)),'lys_7':((515,585),(370,430)),'lys_8':((512,582),(425,485)),
#                 'lys_9':((388,458),(22,82)),'lys_10':((391,461),(84,144)),'lys_11':((390,460),(137,197)),'lys_12':((404,474),(187,247)),
#                 'lys_13':((411,481),(265,325)),'lys_14':((403,473),(320,380)),'lys_15':((410,470),(381,441)),'lys_16':((407,467),(431,491)),
#                 'lys_17': ((184, 254), (38, 98)), 'lys_18': ((180, 250), (94, 154)), 'lys_19': ((180, 250), (151, 211)),
#                 'lys_20':((185,255),(207,267)),'lys_21':((192,262),(276,336)),'lys_22':((190,260),(335,395)),'lys_23':((184,254),(386,446)),
#                 'lys_24':((190,260),(440,500)),'lys_25':((61,131),(41,101)),'lys_26':((63,133),(196,156)),'lys_27':((56,126),(145,205)),
#                 'lys_28':((67,137),(203,263)),'lys_29':((63,133),(278,338)),'lys_30':((65,135),(335,395)),'lys_31':((67,137),(388,448)),
#                 'lys_32':((69,139),(448,508))}
#
exp3_pixelmap= {'lys_1':((508,578),(26,76)),'lys_2':((512,582),(75,135)),'lys_3':((500,570),(129,189)),'lys_4':((508,578),(184,244)),
                'lys_5':((516,586),(264,324)),'lys_6':((516,586),(318,378)),'lys_7':((515,585),(370,430)),'lys_8':((512,582),(425,485)),
                'lys_9':((388,458),(22,82)),'lys_10':((391,461),(84,144)),'lys_11':((390,460),(137,197)),'lys_12':((404,474),(187,247)),
                'lys_13':((411,481),(265,325)),'lys_14':((403,473),(320,380)),'lys_16':((407,467),(431,491)),
                'lys_20':((185,255),(207,267)),'lys_21':((192,262),(276,336)),'lys_22':((190,260),(335,395)),'lys_23':((184,254),(386,446)),
                'lys_24':((190,260),(440,500)),'lys_25':((61,131),(41,101)),'lys_26':((63,133),(196,156)),'lys_27':((56,126),(145,205)),
                'lys_28':((67,137),(203,263)),'lys_30':((65,135),(335,395)),'lys_31':((67,137),(388,448)),
                'lys_32':((69,139),(448,508))}

# 'lys_17': ((184, 254), (38, 98)), 'lys_18': ((180, 250), (94, 154)), 'lys_19': ((180, 250), (151, 211)),

pixelmap= {'exp1000':exp1000_pixelmap,'exp3':exp3_pixelmap}

exp1000_norm_map= {'dry':((306,324),(394,408)),'wet':((584,594),(210,220))}
exp3_norm_map= {'dry':((305,334),(223,260)),'wet':((208,225),(186,199))}

norm_map= {'exp3':exp3_norm_map,'exp1000':exp1000_norm_map}

# exp1000_dataloc= "//Users/roiklein/Dropbox/Msc Project/Deep Learning Project/Exp1000_Full/"
# exp1000_labelloc= '/Users/roiklein/Dropbox/Msc Project/Deep Learning Project/lys_prc.csv'
#
# exp3_dataloc= '/Users/roiklein/Dropbox/Exp3_5/'
# exp3_labelloc= '/Users/roiklein/Desktop/PNN/ET_3-5_hourly.csv'
# exp3_labelloc= '/Users/roiklein/Desktop/PNN/ET_3-5_version1.csv'
#
exp1000_dataloc = expanduser('~/Exp1000/')
exp1000_labelloc = expanduser('~/PNN-AI/lys_prc.csv')

exp3_dataloc= expanduser('~/Exp3_5/')
exp3_labelloc= expanduser('~/Exp3.5_OldCode/ET_3-5_version1.csv')
#
# SAVE_DIR = expanduser('~/PNN-AI/')

dataloc_dict = dict(exp1000= exp1000_dataloc,exp3= exp3_dataloc)
labelloc_dict = dict(exp1000=exp1000_labelloc,exp3= exp3_labelloc)

list_of_exp= ['exp1000','exp3']
list_of_keys= [['lys1','lys2','lys3','lys4','lys5','lys6','lys7','lys8','lys9','lys10'],
               ['lys_1','lys_2','lys_3','lys_4','lys_5','lys_6','lys_7','lys_8','lys_9','lys_10',
                'lys_11','lys_12','lys_13','lys_14','lys_15','lys_16','lys_17','lys_18','lys_19','lys_20',
                'lys_21','lys_22','lys_23','lys_24','lys_25','lys_26','lys_27','lys_28','lys_29','lys_30',
                'lys_31','lys_32']]



exp_args= dict(num_epochs= 500, lr= 2e-4, embedding_dim= 2048,tcn_num_levels= 5, tcn_hidden_channels= 1024,
           tcn_kernel_size=3 ,tcn_dropout= 0.3,TimeWindow= 8,batch_size=8, start_date = '2020-03-09-00-00-00', end_date= '2020-03-26-00-00-00')

# line_dict= {'exp3':{'line1':['lys_7','lys_13','lys_20','lys_22','lys_28'],'line2':['lys_4','lys_12','lys_14','lys_23','lys_30'],
#                     'line3':['lys_1','lys_10','lys_16','lys_25','lys_31'],'line4':['lys_3','lys_6','lys_21','lys_26','lys_32'],
#                     'line5':['lys_2','lys_8','lys_9','lys_24','lys_27'],'line6':['lys_5','lys_11','lys_15','lys_29']},
#             'exp1000':{'line7':['lys1','lys2','lys3','lys4','lys5','lys6','lys7','lys8','lys9','lys10']}}

line_dict= {'exp3':{'line1':['lys_7','lys_20','lys_22','lys_28'],'line2':['lys_4','lys_23','lys_30'],
                    'line3':['lys_1','lys_10','lys_16','lys_25','lys_31'],'line4':['lys_3','lys_6','lys_21','lys_26','lys_32'],
                    'line5':['lys_2','lys_9','lys_24','lys_27'],'line6':['lys_5','lys_29']},
            'exp1000':{'line7':['lys1','lys2','lys3','lys4','lys5','lys6','lys7','lys8','lys9','lys10']}}
