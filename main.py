import torch.nn as nn
import albumentations as A
from utils import utils
from utils.dataset import Dataset
from utils.scheduler import Scheduler
from models.resnet import ResNet18
import warnings

# Initialize
utils.set_seed(1)
warnings.filterwarnings('ignore')

# Configurable parameters
batch_size = 512
num_epochs = 20
kwargs = {'lr': 0.01, 'momentum': 0.9}
label_mapper = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
device = utils.get_device()

dataset = Dataset('CIFAR10', batch_size=batch_size, trans_lib='albumentations')
# Add augmentations for training dataset
dataset.dt.set_transforms('train', [
    A.HorizontalFlip(),
    A.PadIfNeeded(min_height=40, min_width=40),
    A.RandomCrop(height=32, width=32),
    A.CoarseDropout(max_holes=1, max_height=16, max_width=16, fill_value=dataset.dt.mean)
], True)

# Setting up the parameters for training
model = ResNet18().to(device)
train_loader, test_loader = dataset.get_data_loaders()
criterion = nn.CrossEntropyLoss()
optimizer = utils.get_optimizer('SGD', model, **kwargs)
max_lr = utils.get_best_lr(model, train_loader, criterion, optimizer)
steps_per_epoch = len(train_loader)

scheduler = Scheduler('OneCycleLR', {
    'optimizer': optimizer,
    'max_lr': max_lr,
    'steps_per_epoch': steps_per_epoch, 
    'epochs': num_epochs, 
    'pct_start': 5/num_epochs,
    'div_factor': 100,
    'final_div_factor': 100,
    'three_phase': False,
    'anneal_strategy': 'linear',
    'verbose': False
})

# Training
for epoch in range(1, num_epochs+1):
    print('EPOCH:', epoch)
    model.model_train(device, train_loader, criterion, optimizer, scheduler)
    model.model_test(device, test_loader, criterion)

# Visualizing the model's output with GradCAM
images, actual, pred = model.get_incorrect_pred(device, test_loader)
images, actual, pred = images.to('cpu'), actual.to('cpu'), pred.to('cpu')
labels = [str(indx + 1) + ' Actual: %s' % label_mapper[act.item()] + '\n' + \
    'Pred: %s' % label_mapper[pr.item()] for indx, (act, pr) in enumerate(zip(actual, pred))]
utils.visualize_imgs_with_gradcam(model, images, labels, pred, dataset.dt.mean, dataset.dt.std, 
    n_cols=4, figsize=(12, 15), img_title='Model predictions with GradCAM')
