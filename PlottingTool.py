from matplotlib import pyplot as plt
from Configuration import SAVE_DIR
import os
import torch
import numpy as np
def PlotResults():
    LoadPath= f'{os.getcwd()}/Results.pt'
    LoadPath2= f'{os.getcwd()}/Results2.pt'

    saved_results1= torch.load(LoadPath)
    train_loss= np.sqrt(saved_results1['train_loss']); val_loss= np.sqrt(saved_results1['val_loss'])
    test_loss= np.sqrt(saved_results1['test_loss'])

    saved_results2 = torch.load(LoadPath2)
    train_loss2 = np.sqrt(saved_results2['train_loss']); val_loss2 = np.sqrt(saved_results2['val_loss'])
    test_loss2 = np.sqrt(saved_results2['test_loss'])
    Epochs= range(len(train_loss))

    plt.figure()
    plt.title(f'Using Lines - {1,2,3,4} as Train and test on Lines - {5}\n Test Loss:{test_loss}')
    plt.plot(Epochs,train_loss,label='train loss')
    plt.plot(Epochs,val_loss,label='val loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.grid()

    plt.figure()
    plt.title(f'Using Lines - {1, 2, 3, 4,5} as Train and test on Exp1000\n Test Loss:{test_loss2}')
    plt.plot(Epochs, train_loss2, label='train loss')
    plt.plot(Epochs, val_loss2, label='val loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.grid()


if __name__ == '__main__':
    PlotResults()
    plt.show()