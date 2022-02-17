import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import time
import sys
import os


torch.manual_seed(1)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'data'
dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
         for x in ['train', 'test']}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=4,
                                               shuffle=True, num_workers=4)
                for x in ['train', 'test']}
dset_sizes = {x: len(dsets[x]) for x in ['train', 'test']}
dset_classes = dsets['train'].classes


# Learning Parameters
training_epochs = 25
batch_size = 100
learning_rate = 0.1

fig_accuracy = []
fig_loss     = []
last_epoch_overfit = -1
overfit_counter    = 0


# Network Parameters


# Data Loader
train_loader = dset_loaders['train']
test_loader = dset_loaders['test']

# Define The Model
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,
        	                   out_channels=16,
            				   kernel_size=(5,5),
            				   stride=(1,1))
        self.conv2 = nn.Conv2d(in_channels=16,
            				   out_channels=64,
            				   kernel_size=(5,5),
            				   stride=(1,1))
        self.fc1 = nn.Linear(in_features=5*5*64,
            			     out_features=256)
        self.fc2 = nn.Linear(in_features=256,
            			     out_features=10)

    def forward(self,x):

        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*5*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        
    def name(self):
        return 'lenet'
   
#useful function    
def overfitting(accuracy):
    status    = False
    threshold = 0.01
    if(len(accuracy) > 1):
        change_acc = accuracy[-1] - accuracy[-2]
        if(change_acc <= threshold):
            status = True
    return status

# Create The Model and Optimizer
model = ConvNet()
print(model)
ceriation = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training Epochs
start_process = time.time()
for epoch in range(training_epochs):
    start_epoch = time.time()
    avg_loss = 0
    train_accuracy = 0
    
    # Training
    c = 0
    for x, target in train_loader:
        
        optimizer.zero_grad()
        x, target = Variable(x), Variable(target)
        out = model(x)
        loss = ceriation(out, target)
        loss.backward()
        optimizer.step()
        c+=1
        
        
        avg_loss += loss.data[0]/len(train_loader)
        _, pred_label = torch.max(out.data, 1)
        
        train_accuracy += (pred_label == target.data).sum()
    train_accuracy1 = train_accuracy / len(train_loader)
   # print('ac=', train_accuracy1, train_accuracy, len(train_loader))
   # sys.exit()
    end_epoch = time.time()
    
    print("Epoch:", epoch+1, "Train Loss:", avg_loss,
          "Train Accuracy", train_accuracy,
          "in:", int(end_epoch-start_epoch), "sec")
    
    fig_loss.append(avg_loss)
    
    # Test
    test_accuracy = 0
    for x, target in test_loader:
       x, target = Variable(x, volatile=True),\
                    Variable(target, volatile=True)
       out= model(x)
       _, pred_label = torch.max(out.data, 1)
       test_accuracy += \
       (pred_label == target.data).sum() / len(test_loader)
    print("Test Accuracy", test_accuracy)
    
    fig_accuracy.append(test_accuracy)
    #check overfitting
    if (overfitting(fig_accuracy) & (last_epoch_overfit + 1 == epoch)):
        overfit_counter    += 1
        last_epoch_overfit = epoch
    else:
        overfit_counter = 0
    if(overfit_counter == 3):
        break
    
end_process = time.time()
print ("Train (& test) completed in:", int(end_process-start_process), "sec")

if(epoch != training_epochs):
    print("Overfitting is occurred in epoch=", epoch+1, "so early stopping")
fig_epoch = np.arange(1, epoch+2)

plt.figure(1)
plt.plot( fig_epoch, fig_accuracy)
plt.xlabel('Epoch')
plt.title('Accuracy (%) on the test data')
plt.grid(True)
plt.savefig("accuracy.png")

plt.figure(2)
plt.plot( fig_epoch, fig_loss)
plt.xlabel('Epoch')
plt.title('Loss (%) on the train data')
plt.grid(True)
plt.savefig("loss.png")

plt.show()
