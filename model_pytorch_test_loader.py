#time tracking
import time 

tmps = time.time()
def tmp():
	global tmps
	x = time.time() - tmps
	tmps = time.time()
	return x

#Import
import torch
from torch.utils import data
import torchvision as tv
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

print(" import time : %f" %(tmp()))

#Device set
if torch.cuda.is_available():
	device=torch.device("cuda:3")
else:
	device=torch.device("cpu")
print("using device %s"%(device))


#Data_set
transform = tv.transforms.Compose([tv.transforms.Grayscale(num_output_channels=1),
	tv.transforms.ToTensor(),
	tv.transforms.Normalize((0.5), (0.5))])
dataset = tv.datasets.ImageFolder('/grid_mnt/data__data.polcms/cms/sghosh/camdata/Augmented_dataset_bin/', transform=transform)

classes = ('valid', 'invalid')

batch_size = 6

loader=torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)

print(" loader setup time : %f" %(tmp()))


img_height = 240
img_width = 470
num_classes = 2

#model
class Sequential(nn.Module):
	def __init__(self):
		super(Sequential,self).__init__()
		self.conv1=nn.Conv2d(1, 16, 3)
		self.conv2=nn.Conv2d(16, 32, 3)
		self.conv3=nn.Conv2d(32, 64, 3)
		#self.conv4=nn.Conv2d(64, 128, 3)
		self.pool=nn.MaxPool2d(2, 2)
		#self.lin_size = 64*(img_height//8)*(img_width//8) #supposé juste...
		self.lin_size = 102144
		self.fc1=nn.Linear(self.lin_size, 256)
		self.fc2=nn.Linear(256, 128)
		self.fc3=nn.Linear(128, 32)
		self.fc4=nn.Linear(32, num_classes)
	def forward(self, x):
		x=self.pool(F.relu(self.conv1(x)))
		x=self.pool(F.relu(self.conv2(x)))
		x=self.pool(F.relu(self.conv3(x)))
		#x=self.pool(F.relu(self.conv4(x)))
		x=x.view(x.size(0), -1)
		x=F.relu(self.fc1(x))
		x=F.relu(self.fc2(x))
		x=F.relu(self.fc3(x))
		return self.fc4(x)

model=torch.load("mymodel_robot.pth")
model.to(device)

print(" model setup time : %f" %(tmp()))

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(loader)
images, labels = dataiter.next()

# print images
imshow(tv.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

images, labels = images.to(device), labels.to(device)
outputs = model(images)

# sm = nn.Softmax(dim=1) 
# sm_outputs = sm(outputs) 
# print(sm_outputs)

_, predicted = torch.max(outputs, 1)


print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(batch_size)))