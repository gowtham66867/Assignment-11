# Library imports
import torch
import torch.nn as nn
from torchinfo import summary
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.utils import fabric

class Model_Composite(nn.Module):
    """ This class is created based on the composite design principle. It contains all the common methods related to the model.
    """

    def __init__(self):
        """ Constructor to initialize train and test related properties
        """
        super(Model_Composite, self).__init__()
        self.train_accuracy = []
        self.test_accuracy = []
        self.train_losses = []
        self.test_losses = []
        self.norm_type = 'bn'
        self.n_groups = 2

    def print_summary(self, input_size: tuple):
        """ Print model summary

        Args:
            input_size: Input image size in (N, C, H, W)
        """
        summary(self, input_size, verbose=1)

    def model_train(self, device, train_loader, criterion, optimizer, scheduler=None) -> None:
        """ Training the model

        Args:
            device: CUDA/CPU
            train_loader: train data loader instance of `torch.utils.data.DataLoader`
            criterion: loss function
            optimizer: optimizer function
        """
        self.train()
        pbar = tqdm(train_loader)
        train_loss = 0
        correct = 0
        processed = 0
        for data, target in pbar:
            data, target = data.to(device), target.to(device) # comment this line if using fabric
            
            # Init
            optimizer.zero_grad()
            
            # Prediction
            y_pred = self(data)
            
            # Calculate loss for a batch of images
            loss = criterion(y_pred, target)
            train_loss += loss.item()

            pred = y_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)

            # Backpropagation
            loss.backward()
            # fabric.backward(loss)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            pbar.set_description(desc='Train: Loss={:0.4f}, LR={}, Accuracy={:0.2f}'.format(
                loss.item(),
                optimizer.param_groups[0]["lr"],
                100*correct/processed
            ))
        
        self.train_accuracy.append(100*correct/processed)
        self.train_losses.append(train_loss)

    def model_test(self, device, test_loader, criterion) -> None:
        """ Testing the model

        Args:
            device: CUDA/CPU
            test_loader: test data loader instance of `torch.utils.data.DataLoader`
            criterion: loss function
        """
        self.eval()
        test_loss = 0
        correct = 0
        processed = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                y_pred = self(data)
                try:
                    test_loss += criterion(y_pred, target, reduction='sum').item()
                except:
                    test_loss += criterion(y_pred, target).item()
                pred = y_pred.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                processed += len(data)

        self.test_losses.append(test_loss)
        # To compute the average test loss, divide the total loss by number of iterations
        test_loss /= (len(test_loader.dataset) / test_loader.batch_size)
        
        self.test_accuracy.append(100. * correct / len(test_loader.dataset))
        print('Test: Average Loss: {:0.4f}, Accuracy: {}/{} ({:.2f}%), Accuracy Diff: {}\n'.format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100. * correct / len(test_loader.dataset),
            abs(round(self.test_accuracy[-1] - self.train_accuracy[-1], 4))
        ))

    def get_norm(self, n_channels: int):
        """ Get a normalization layer. It could be one of the following:
            - Batch normalization
            - Layer normalization
            - Group normalization

        Args:
            n_channels: number of channels to create a normalization layer
        """
        if self.norm_type == 'bn':
            return nn.BatchNorm2d(n_channels)
        elif self.norm_type == 'ln':
            return nn.GroupNorm(1, n_channels)
        elif self.norm_type == 'gn':
            return nn.GroupNorm(self.n_groups, n_channels)

    def plot_accuracy(self) -> None:
        """
        Visualize model train and test accuracy over n_epochs
        """
        epochs = list(range(1, len(self.train_accuracy) + 1))
        plt.plot(epochs, self.train_accuracy, label='Train accuracy')
        plt.plot(epochs, self.test_accuracy, label='Test accuracy')
        plt.legend()

    def plot_loss(self) -> None:
        """
        Visualize model train and test losses over n_epochs
        """
        epochs = list(range(1, len(self.train_accuracy) + 1))
        plt.plot(epochs, self.train_losses, label='Train loss')
        plt.plot(epochs, self.test_losses, label='Test loss')
        plt.legend()

    def get_incorrect_pred(self, device: str, test_loader, top_n: int=10) -> tuple:
        """ Get top_n incorrect model predictions from test set.

        Args:
            device: CUDA/CPU
            test_loader: test data loader instance of `torch.utils.data.DataLoader`
            top_n: number of incorrect predictions to return
        """
        with torch.no_grad():
            data, target = next(iter(test_loader))
            data, target = data.to(device), target.to(device)
            y_pred = self(data)
            pred = y_pred.argmax(dim=1)
            compare = pred.eq(target)
            incorrect_indx = torch.where((compare == False), 1, 0).nonzero()
            top_n_pred = incorrect_indx[:top_n].squeeze()
            return (data[top_n_pred], target[top_n_pred], pred[top_n_pred])
