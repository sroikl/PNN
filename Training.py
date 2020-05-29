import torch
import copy
import tqdm
import numpy as np

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=25):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e5
    train_loss_list, val_lost_list= [],[]
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 25)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                # scheduler.step()
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = []

            # Iterate over data.
            with tqdm.tqdm(total=len(dataloaders[phase]),desc=f'{phase}') as pbar:
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.transpose(0,1).double().to(device)
                    labels = labels.transpose(0,1).double().to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss.append(loss.item())
                    pbar.set_description(f'{phase} ({loss.item():.3f})')
                    pbar.update()

                epoch_loss = np.sum(running_loss) /len(dataloaders[phase].batch_sampler)

            print('{} Epoch Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss <= best_loss:
                best_loss= epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'train':
                train_loss_list.append(epoch_loss)
            elif phase == 'val':
                val_lost_list.append(epoch_loss)

    # load best model weights
    # scheduler.step()
    model.load_state_dict(best_model_wts)
    return model,train_loss_list,val_lost_list


def eval_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=25):

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 25)

        for phase in ['test']:
            model.eval()  # Set model to evaluate mode

            running_loss = []

            # Iterate over data.
            with tqdm.tqdm(total=len(dataloaders[phase]),desc=f'{phase}') as pbar:
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.transpose(0, 1).double().to(device)
                    labels = labels.transpose(0, 1).double().to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    # statistics
                    running_loss.append(loss.item())
                    pbar.set_description(f'{phase} ({loss.item():.3f})')
                    pbar.update()

            epoch_loss = np.sum(running_loss) / len(dataloaders[phase].batch_sampler)

            print('{} Epoch Loss: {:.4f}'.format(phase, epoch_loss))


    return epoch_loss

