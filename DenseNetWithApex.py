import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import torch.utils.checkpoint as cp

import numpy as np
import math
from collections import OrderedDict

# Apex module allows automatic mixed precision on floats to speed up training process while minimising accuracy loss
try:
    # from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


NUM_TRAIN = 49000

# The torchvision.transforms package provides tools for preprocessing data
# and for performing data augmentation; here we set up a transform to
# preprocess the data by subtracting the mean RGB value and dividing by the
# standard deviation of each RGB value; we've hardcoded the mean and std.
transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

# We set up a Dataset object for each split (train / val / test); Datasets load
# training examples one at a time, so we wrap each Dataset in a DataLoader which
# iterates through the Dataset and forms minibatches. We divide the CIFAR-10
# training set into train and val sets by passing a Sampler object to the
# DataLoader telling how it should sample from the underlying Dataset.
cifar10_train = dset.CIFAR10('./datasets', train=True, download=True,transform=transform)
loader_train = DataLoader(cifar10_train, batch_size=16, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

cifar10_val = dset.CIFAR10('./datasets', train=True, download=True,transform=transform)
loader_val = DataLoader(cifar10_val, batch_size=16, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

cifar10_test = dset.CIFAR10('./datasets', train=False, download=True, transform=transform)
loader_test = DataLoader(cifar10_test, batch_size=16)

USE_GPU = True

dtype = torch.float32 # we will be using float throughout 

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss
print_every = 100

print('using device:', device)

def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


def train(model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            # loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy(loader_val, model)
                print()


class DenseLayer(nn.Module):
    def __init__(self,input_size,growth_rate):
        super().__init__()
        self.input_size = input_size
        self.growth_rate = growth_rate
        bottle_neck_k = 4*self.growth_rate

        self.bn1 = nn.BatchNorm2d(self.input_size)
        self.c1x1 = nn.Conv2d(in_channels=self.input_size, out_channels=bottle_neck_k, kernel_size=1,stride=1,padding=0,bias=False)             #Bottleneck
        nn.init.kaiming_normal_(self.c1x1.weight)
        self.bn2 = nn.BatchNorm2d(bottle_neck_k)
        self.c3x3 = nn.Conv2d(in_channels=bottle_neck_k,out_channels=self.growth_rate,kernel_size=3,stride=1,padding=1,bias=False)
        nn.init.kaiming_normal_(self.c3x3.weight)
        
    def forward(self, x):
        self.x = x
        out1 = self.c1x1(F.relu(self.bn1(self.x)))
        out2 = self.c3x3(F.relu(self.bn2(out1)))
        return torch.cat((self.x,out2),dim=1)

class TrasitionBlock(nn.Module):
    def __init__(self,in_channels,dropout):
        super().__init__()
        self.in_channels = in_channels
        self.dropout = dropout
        out_channels = int(in_channels*self.dropout)
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.conv = nn.Conv2d(self.in_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self,x):
        self.x = x
        out = self.conv(F.relu(self.bn(self.x)))
        return F.avg_pool2d(out,2,stride=2)

class DenseNet(nn.Module):
    def __init__(self,input_channels = 3,growth_rate=12,num_classes=10,dropout_ratio=0.5,num_dense_layers=16):
        super().__init__()
        
        self.input_channels = input_channels
        self.dropout_ratio = dropout_ratio
        self.growth_rate = growth_rate
        self.num_dense_layers = num_dense_layers
        self.num_classes = num_classes

        c1_out = 2*self.growth_rate
        t1_in = int(c1_out + self.num_dense_layers*self.growth_rate)
        t1_out = int(t1_in*self.dropout_ratio)
        t2_in = int(t1_out + self.num_dense_layers*self.growth_rate)
        t2_out = int(t2_in*self.dropout_ratio)
        d3_out = int(t2_out + self.num_dense_layers*self.growth_rate)

        self.c1_bn = nn.BatchNorm2d(self.input_channels)
        self.c1 = nn.Conv2d(in_channels=self.input_channels, out_channels=c1_out, kernel_size=3,stride=1,padding=1,bias=False)
        nn.init.kaiming_normal_(self.c1.weight)
        self.d1 = self.make_dense_block(c1_out,self.num_dense_layers,self.growth_rate)
        self.t1 = TrasitionBlock(t1_in,self.dropout_ratio)
        self.d2 = self.make_dense_block(t1_out,self.num_dense_layers,self.growth_rate)
        self.t2 = TrasitionBlock(t2_in,self.dropout_ratio)
        self.d3 = self.make_dense_block(t2_out,self.num_dense_layers,self.growth_rate)
        
        self.classifier = nn.Linear(d3_out, num_classes)

    def forward(self,x):
        self.x = x
        out1 = self.c1(F.relu(self.c1_bn(self.x)))
        out2 = self.t1(self.d1(out1))
        out3 = self.t2(self.d2(out2))
        out4 = self.d3(out3)
        out = F.adaptive_avg_pool2d(out4, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out) #Softmax is done inside of Pytorch
        return out

    def make_dense_block(self, nInput,nLayers, growthRate):
        self.nInput = nInput
        self.nLayers = nLayers
        self.growthRate = growthRate
        layers = []
        for i in range(int(self.nLayers)):
            layers.append(DenseLayer(self.nInput,self.growthRate))
            self.nInput += self.growthRate
        return nn.Sequential(*layers) #can use instead of nn.ModuleList

if __name__ == "__main__":
    learning_rate = 0.1
    model = DenseNet()
    model = model.to(device=device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9, nesterov=True)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    train(model, optimizer, epochs=10)