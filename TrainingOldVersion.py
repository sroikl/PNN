from torch.utils.data import DataLoader
import abc
import tqdm
import torch
import os
import numpy as np
from Configuration import exp_args

class Trainer:
    def __init__(self,SAVE_DIR):
        self.save_dir= os.path.join(SAVE_DIR,'Results.pt')

    def fit(self,dl_train:DataLoader,dl_test:DataLoader,num_epochs:int,**kwargs):

        loss_train,loss_test = [],[]

        for Epoch in range(num_epochs):

            print(f'--- EPOCH {Epoch + 1}/{num_epochs} ---')

            loss_tr_epoch = self._train_epoch(dl_train,Epoch,**kwargs)
            loss_train.append(np.mean(loss_tr_epoch))

            loss_ts_epoch = self._test_epoch(dl_test,Epoch,**kwargs)
            loss_test.append(np.mean(loss_ts_epoch))


        #TODO: add features such as checkpoints, early stopping etc.


    def _train_epoch(self,dl_train,Epoch,**kwargs):
        return self._ForBatch(dl_train,self.train_batch,Epoch,**kwargs)
    def _test_epoch(self,dl_test,Epoch,**kwargs):
        return self._ForBatch(dl_test, self.test_batch,Epoch, **kwargs)

    @abc.abstractmethod
    def train_batch(self, batch,Epoch_num):
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch,Epoch_num):
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _ForBatch(dl:DataLoader,forward_fn,Epoch):

        losses = []
        num_batches = len(dl.batch_sampler)
        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                p= float(batch_idx + Epoch*num_batches)/(exp_args['num_epochs'] * num_batches)
                grl_lambda= 2./(1.+np.exp(-10*p)) - 1
                if Epoch % 10 == 0:
                    save_model= True

                batch_loss,Transpiration_loss,Plant_loss = forward_fn(data,grl_lambda,save_model)

                pbar.set_description(f'{pbar_name} ({batch_loss:.3f}),Trans Loss({Transpiration_loss:.3f}),Plant Loss({Plant_loss:.3f}),grl_lambda({grl_lambda:.3f})')
                pbar.update()

                losses.append(batch_loss)

            avg_loss = sum(losses) / num_batches
            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Loss {avg_loss:.3f},')

        return losses

class TCNTrainer(Trainer):

    def __init__(self, model,Transpiration_loss_fn,Plant_loss_fn,optimizer ,num_plants,device=None):
        super(Trainer, self).__init__()

        self.model = model
        self.Transpiration_loss_fn = Transpiration_loss_fn
        self.Plant_loss_fn= Plant_loss_fn
        self.optimizer = optimizer
        self.device = device
        self.num_plants= num_plants
    def train_batch(self, batch,grl_lambda,save_model=False):
        X,y_tr,y_pl = batch

        #Shuffle across the num plants dimension
        idx= torch.randperm(self.num_plants)
        X=X[:,:,idx,:,:] ; y_tr= y_tr[:,:,idx] ; y_pl= y_pl[:,:,idx]


        x = X.squeeze(dim=0).transpose(0, 1).unsqueeze(dim=2).repeat(1,1,3,1,1).to(self.device)
        y_tr = y_tr.squeeze(dim=0).transpose(0, 1).to(self.device)
        y_pl = y_pl.squeeze(dim=0).to(self.device)

        self.optimizer.zero_grad()
        #Forward Pass
        Transpiration_predictions,Plant_predictions= self.model(x,grl_lambda= grl_lambda)

        # Compute Loss
        transpiration_loss= self.Transpiration_loss_fn(Transpiration_predictions,y_tr)
        plant_loss= self.Plant_loss_fn(Plant_predictions.transpose(0,1),y_pl)

        loss= transpiration_loss + plant_loss
        #Back Prop
        loss.backward()

        #Update params
        self.optimizer.step()

        if save_model:
            torch.save(self.model.state_dict(),f'{os.getcwd()}/Model.pt')

        return loss.item(),transpiration_loss.item(),plant_loss.item()

    def test_batch(self, batch,grl_lambda):
        X, y_tr, y_pl = batch
        x = X.squeeze(dim=0).transpose(0, 1).unsqueeze(dim=2).repeat(1, 1, 3, 1, 1).to(self.device)
        y_tr = y_tr.squeeze(dim=0).transpose(0, 1).to(self.device)
        y_pl = y_pl.squeeze(dim=0).transpose(0, 1).to(self.device)

        with torch.no_grad():
            # Forward Pass
            Transpiration_predictions, Plant_predictions = self.model(x, grl_lambda=grl_lambda)

            #  # Compute Loss
            transpiration_loss = self.Transpiration_loss_fn(Transpiration_predictions, y_tr)
            plant_loss = self.Plant_loss_fn(Plant_predictions, y_pl)

            loss = transpiration_loss + plant_loss

        return loss.item(),transpiration_loss.item(),plant_loss.item()