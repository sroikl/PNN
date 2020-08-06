import torch
import numpy as np
import os
import tqdm
from Model import TemporalSpatialModel
import matplotlib.pyplot as plt
from DataLoader import DataLoader
data_path = '/Users/roiklein/Dropbox/Exp3_5'
model_path = f'{os.getcwd()}/Model_Exp3.5.pt'
from Main import Split_Test_Train
from Configuration import  exp_args,exp3_pixelmap
def Inference():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # UseExp1000= True
    UseExp1000= False

    print('==== Loading Data ====')
    if UseExp1000:
        data = DataLoader(UseExp3_5= False)
        X, y = data.LoadData()
        validation_split = 0.25
        dataset = torch.utils.data.TensorDataset(X, y)

        dl_train, dl_test = Split_Test_Train(dataset=dataset,validation_split=validation_split,batch_size=8,shuffle_test=False)
    else:
        data = DataLoader(UseExp3_5=True)
        X, y = data.LoadData()

        validation_split = 0.4
        dataset = torch.utils.data.TensorDataset(X, y)

        dl_train, dl_test = Split_Test_Train(dataset=dataset, validation_split=validation_split, batch_size=4,
                                             shuffle_test=False)
        print(f'Sampels Shape is:%s' % {next(iter(dl_train))[0].shape})
        print(f'Label Shape is:%s' % {next(iter(dl_train))[1].shape})
        names= list(exp3_pixelmap.keys())

    print('==== Loading Model ====')
    model = TemporalSpatialModel(num_levels=exp_args['tcn_num_levels'], num_hidden=exp_args['tcn_hidden_channels'],
                                 embedding_size=exp_args['embedding_dim'], kernel_size=exp_args['tcn_kernel_size'],
                                 dropout=exp_args['tcn_dropout']).to(device=device)

    model.load_state_dict(torch.load(model_path))
    y_true,y_pred= [],[]
    for x,y in tqdm.tqdm(dl_test,desc='Inference'):
        output= model(x.squeeze(dim=0).transpose(0, 1)).detach().numpy()
        y_pred.append(output)
        y_true.append(y.squeeze(dim=0).transpose(0, 1).detach().numpy())

    data_true= np.concatenate((y_true[0],y_true[1]),axis=1)
    data_pred= np.concatenate((y_pred[0],y_pred[1]),axis=1)

    for i in range(2,len(y_true)):
        data_true= np.concatenate((data_true,y_true[i]),axis=1)
    for i in range(2,len(y_pred)):
        data_pred= np.concatenate((data_pred,y_pred[i]),axis=1)

    num_plants,data_points= np.shape(data_true)
    for i in range(num_plants):
        plt.figure()
        plt.title(f'{names[i]} Transpiration')
        plt.plot(data_true[i,:],label= 'True Transpiration')
        plt.plot(data_pred[i,:],'o',label='Predicted Transpiration')
        plt.legend(); plt.grid()
    plt.show()
if __name__ == '__main__':
    Inference()


