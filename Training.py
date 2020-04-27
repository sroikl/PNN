import torch
import time
import copy
import tqdm

def train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e5

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 25)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            with tqdm.tqdm(total=len(dataloaders[phase]),desc=f'{phase}') as pbar:
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.double().to(device)
                    labels = labels.double().to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs.squeeze())
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    pbar.update()

                epoch_loss = running_loss / (len(labels)*len(dataloaders[phase]))

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss <= best_loss:
                best_loss= epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def eval_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=25):
    since = time.time()


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 25)

        for phase in ['test']:
            model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            with tqdm.tqdm(total=len(dataloaders[phase]),desc=f'{phase}') as pbar:
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs.squeeze())
                        loss = criterion(outputs, labels)

                    # statistics
                    running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / (len(labels) * len(dataloaders[phase]))

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Loss: {:4f}'.format(epoch_loss))
    return epoch_loss

