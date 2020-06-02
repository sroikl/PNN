from matplotlib import pyplot as plt
import matplotlib
from Configuration import SAVE_DIR
import os
import torch
import numpy as np
import itertools
def PlotResults():
    LoadPath= '/Users/roiklein/Desktop/PNN/OldVersionCode/Results.pt'
    LoadPath2= f'{os.getcwd()}/Results2.pt'

    saved_results1= torch.load(LoadPath)
    train_loss= np.sqrt(saved_results1['loss_train']);
    test_loss= np.sqrt(saved_results1['loss_test']);

    Epochs= range(len(train_loss))


    plt.figure()
    plt.plot(Epochs,train_loss,label='train loss')
    plt.plot(Epochs,test_loss,label='val loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.grid()



if __name__ == '__main__':
    PlotResults()
    plt.show()