import argparse
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

exp3_pixelmap= {'lys_1':((518,571),(26,76)),'lys_2':((520,574),(79,131)),'lys_3':((505,571),(131,187)),'lys_4':((512,574),(188,240)),
                'lys_5':((521,582),(265,323)),'lys_6':((519,582),(323,373)),'lys_7':((519,581),(374,426)),'lys_8':((521,573),(427,483)),
                'lys_9':((389,458),(19,85)),'lys_10':((393,459),(87,141)),'lys_11':((394,460),(142,193)),'lys_12':((413,466),(191,243)),
                'lys_13':((418,475),(266,324)),'lys_14':((410,467),(325,380)),'lys_15':((410,465),(385,437)),'lys_16':((410,464),(438,484)),
                'lys_17':((190,247),(40,95)),'lys_18':((187,243),(100,148)),'lys_19':((183,245),(154,207)),'lys_20':((185,256),(211,263)),
                'lys_21':((195,259),(274,339)),'lys_22':((190,255),(339,391)),'lys_23':((191,248),(392,440)),'lys_24':((196,253),(442,498)),
                'lys_25':((69,124),(43,99)),'lys_26':((73,123),(100,151)),'lys_27':((65,117),(149,202)),'lys_28':((74,131),(202,263)),
                'lys_29':((68,127),(280,335)),'lys_30':((76,123),(339,390)),'lys_31':((72,131),(390,446)),'lys_32':((71,137),(448,508))}

pixelmap= {'exp1000':exp1000_pixelmap,'exp3':exp3_pixelmap}

exp1000_norm_map= {'dry':((306,324),(394,408)),'wet':((584,594),(210,220))}
exp3_norm_map= {'dry':((305,334),(223,260)),'wet':((208,225),(186,199))}

norm_map= {'exp3':exp3_norm_map,'exp1000':exp1000_norm_map}

exp1000_dataloc= "//Users/roiklein/Dropbox/Msc Project/Deep Learning Project/Exp1000_Full/"
exp1000_labelloc= '/Users/roiklein/Dropbox/Msc Project/Deep Learning Project/lys_prc.csv'

exp3_dataloc= '/Users/roiklein/Desktop/PNN/Example DB/'
exp3_labelloc= '/Users/roiklein/Desktop/PNN/DL_TC_EXP_3-5.csv'
#
# exp1000_dataloc = expanduser('~/Exp1000/')
# exp1000_labelloc = expanduser('~/PNN-AI/lys_prc.csv')

# exp3_dataloc= expanduser('~/Exp3_5/')
# exp3_labelloc= expanduser('~/PNN/DL_TC_EXP_3-5.csv')

# SAVE_DIR = expanduser('~/PNN-AI/')

dataloc_dict = dict(exp1000= exp1000_dataloc,exp3= exp3_dataloc)
labelloc_dict = dict(exp1000=exp1000_labelloc,exp3= exp3_labelloc)

list_of_exp= ['exp1000','exp3']
list_of_keys= [['lys1','lys2','lys3','lys4','lys5','lys6','lys7','lys8','lys9','lys10'],
               ['lys_1','lys_2','lys_3','lys_4','lys_5','lys_6','lys_7','lys_8','lys_9','lys_10',
                'lys_11','lys_12','lys_13','lys_14','lys_15','lys_16','lys_17','lys_18','lys_19','lys_20',
                'lys_21','lys_22','lys_23','lys_24','lys_25','lys_26','lys_27','lys_28','lys_29','lys_30',
                'lys_31','lys_32']]



exp_args= dict(num_epochs= 500, lr= 1e-4, embedding_dim= 2048,tcn_num_levels= 5, tcn_hidden_channels= 1000,
           tcn_kernel_size=5,tcn_dropout= 0.25,batch_size= 16)

line_dict= {'exp3':{'line1':['lys_7','lys_13','lys_20','lys_22','lys_28'],'line2':['lys_4','lys_12','lys_14','lys_23','lys_30'],
                    'line3':['lys_1','lys_10','lys_16','lys_25','lys_31'],'line4':['lys_3','lys_6','lys_21','lys_26','lys_32'],
                    'line5':['lys_2','lys_8','lys_9','lys_24','lys_27'],'line6':['lys_5','lys_11','lys_15','lys_17','lys_29']},
            'exp1000':{'line7':['lys1','lys2','lys3','lys4','lys5','lys6','lys7','lys8','lys9','lys10']}}
            # 'exp1000':{'line1':[]}}
