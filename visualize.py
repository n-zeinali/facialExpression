from torchvision import datasets, models, transforms
from confusionmeter import ConfusionFigure
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch
import time
import sys
import os

plt.ion()   # interactive mode

# Data augmentation and normalization for training
# Just normalization for validation
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

data_dir = 'data6'
dsets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
         for x in ['train', 'test']}
dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=4,
                                               shuffle=True, num_workers=4)
                for x in ['train', 'test']}
dset_sizes = {x: len(dsets[x]) for x in ['train', 'test']}
dset_classes = dsets['train'].classes


# use_gpu = torch.cuda.is_available()
use_gpu = False

def visualize_model(model):

    #print('dset=', dset_classes)
    confusionObj = ConfusionFigure(dset_classes)
    
    for i, data in enumerate(dset_loaders['test']):
        inputs, labels = data
        #print('size=', inputs.size()[0])
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        #ADD
        confusionObj.add(preds, labels)
    #showfig
    confusionObj.show_figure(title= 'Confusion matrix, without normalization',
                         saving_path = "confuMatrix.png")
    confusionObj.show_figure(normalize = True,
            title= 'Normalized confusion matrix',
            saving_path = "confuMatrixNorm.png")            
    return

def plot_kernels(tensor, num_cols=6):
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    if not tensor.shape[-1]==3:
        print("last dim needs to be 3 to plot")
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        print(tensor[i])
        sys.exit()
        ax1.imshow(tensor[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
    
def visualize_model1(model, num_images=10):
    images_so_far = 0
    fig = plt.figure(figsize=(8,8))

    for i, data in enumerate(dset_loaders['test']):
        inputs, labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = fig.add_subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(dset_classes[preds[j]]))
            im = inputs.data[j][0].cpu().squeeze().numpy()
            ax.imshow(im.reshape(im.shape[0], im.shape[1]))

            if images_so_far == num_images:
                return
    
    
  
#vgg = models.vgg16(pretrained=True)
#mm = vgg.double()
#filters = mm.modules
#body_model = [i for i in mm.children()][0]
#print('shape1=', body_model)
#layer1 = body_model[28]
#tensor = layer1.weight.data.numpy()
#print(tensor.shape)
#plot_kernels(tensor)
#sys.exit()
path = 'model/res6c20e.pkl'
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
# we have 13classes in my task
model_ft.fc = nn.Linear(num_ftrs, 6)
model_ft.fc.weight.data.normal_(mean=0, std=0.01)

if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)


######
#load
######
model_ft.load_state_dict(torch.load(path))
#confusion matrix
visualize_model(model_ft)

#mm = model_ft.double()
#filters = mm.modules
#body_model = [i for i in mm.children()][0]

#layer1 = body_model
#tensor = layer1.weight.data.numpy()
#print('shape=', tensor.shape)

#plot_kernels(tensor)
#visualize_model1(model_ft)
                   