from cProfile import label
from lib2to3.pytree import convert
from turtle import color, forward
import numpy as np
import time
import matplotlib.pyplot as plt



train_images_np=np.load('./Project3_Data/MNIST_train_images.npy')
train_labels_np=np.load('./Project3_Data/MNIST_train_labels.npy')
val_images_np=np.load('./Project3_Data/MNIST_val_images.npy')
val_labels_np=np.load('./Project3_Data/MNIST_val_labels.npy')
test_images_np=np.load('./Project3_Data/MNIST_test_images.npy')
test_labels_np=np.load('./Project3_Data/MNIST_test_labels.npy')


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def CrossEntropy(y_hat,y):
    return -np.dot(y,np.log(y_hat))

class MLP():

    def __init__(self):
        self.W1 = np.random.normal(loc=0,scale=0.1, size = (64,784))
        self.b1 = 0
        self.W2 = np.random.normal(loc=0,scale=0.1, size = (64,10))
        self.b2 = 0
        self.reset_grad()

    def reset_grad(self):
        self.W2_grad = 0
        self.b2_grad = 0
        self.W1_grad = 0
        self.b1_grad = 0

    def forward(self, x):
        self.x = x
        self.W1x = np.dot(self.W1, self.x) 
        self.a1 = self.W1x + self.b1
        self.f1 = sigmoid(self.a1)
        self.W2x = np.dot(self.W2.T, self.f1) 
        self.a2 = self.W2x + self.b2
        self.y_hat = softmax(np.dot(self.W2.T,(sigmoid(self.a1))) + self.b2)

        return self.y_hat

    def update_grad(self,y):
        dA2db2 = 1
        dA2dW2 = self.f1
        dA2dF1 = self.W2
        dF1dA1 = sigmoid(self.f1)
        dA1db1 = 1
        dA1dW1 = self.x

        dLdA2 = (self.y_hat - y)
        dLdW2 = np.dot(dLdA2.reshape(10,1), dA2dW2.reshape(1,64))
     
        dLdb2 = np.dot(dA2db2, dLdA2)
        dLdF1 = np.dot(np.transpose(dLdA2), np.transpose(dA2dF1))
        dLdA1 = dLdF1 * np.transpose(dF1dA1)
        dLdW1 = np.dot(dLdA1.reshape(64,1), dA1dW1.reshape(1,784))
        dLdb1 = np.dot(dA1db1,np.transpose(dLdA1))

        self.W2_grad = self.W2_grad + dLdW2
        self.b2_grad = self.b2_grad + dLdb2
        self.W1_grad = self.W1_grad + dLdW1
        self.b1_grad = self.b1_grad + dLdb1
        

    def update_params(self,learning_rate):
        self.W2 = self.W2 - learning_rate * np.transpose(self.W2_grad)
        self.b2 = self.b2 - learning_rate * self.b2_grad.reshape(-1) 
        self.W1 = self.W1 - learning_rate * self.W1_grad
        self.b1 = self.b1 - learning_rate * self.b1_grad.reshape(-1)


def onehot_convert(vector):
    reshaped = np.reshape(vector, (len(vector), 1))
    vec_ans = np.empty((len(reshaped), 10))

    for index, vals in enumerate(reshaped):
        vec_ans[index] = 0
        vec_ans[index, vals[0]] = 1
    return vec_ans

def train_mlp(myNet, learning_rate, n_epochs, batch_size, num_images, train_onehot, val_onehot):
    train_accuracy = []
    val_accuracy = []

    for iter in range(n_epochs):
        for i in range(num_images):
            myNet.forward(train_images_np[i])
            myNet.update_grad(train_onehot[i])

            if i % batch_size == 0 or i == num_images-1:
                myNet.update_params(learning_rate)
                myNet.reset_grad()
     
        loss = 0
        correct_val = 0
        correct_train = 0

        for i in range(num_images):
            y_hat_train = myNet.forward(train_images_np[i])
            if np.argmax(y_hat_train) == np.argmax(train_onehot[i]):
                correct_train = correct_train + 1
                    
        for i in range(len(val_images_np)):
            y_hat_val = myNet.forward(val_images_np[i])
            index = np.argmax(val_onehot[i])
            loss = loss + CrossEntropy(y_hat_val[index], val_onehot[i][index])

            if np.argmax(y_hat_val) == index:
                correct_val = correct_val + 1

        train_accuracy.append(correct_train / num_images*100)
        val_accuracy.append(correct_val / len(val_images_np)*100)
        epochs_array.append(iter+1)
        print("Loss per epoch: " + str(loss/len(val_images_np)))
        
    return train_accuracy, val_accuracy


train_onehot = onehot_convert(train_labels_np)
val_onehot = onehot_convert(val_labels_np)
test_onehot = onehot_convert(test_labels_np)
myNet=MLP()

if input("Train or Load data (t or l): ") == "t":

    learning_rate=1e-3
    n_epochs=100
    batch_array_count = []
    epochs_array = []
    num_images = 2000
    batch_size = 256

    train_accuracy, val_accuracy = train_mlp(myNet, learning_rate, n_epochs, batch_size,num_images, train_onehot, val_onehot)


    plt.plot(epochs_array, list(train_accuracy),label="Train", color="Red")
    plt.plot(epochs_array, list(val_accuracy), label="Validation", color="Green")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Training and Validation Accuracy on 2000 images")
    plt.legend()
    plt.show()

    
    learning_rate=1e-3
    n_epochs=100
    batch_array_count = []
    epochs_array = []
    num_images = 50000
    batch_size = 256
    '''
    train_accuracy, val_accuracy = train_mlp(myNet, learning_rate, n_epochs, batch_size,num_images, train_onehot, val_onehot)

    
    plt.plot(epochs_array, list(train_accuracy),label="Train", color="Red")
    plt.plot(epochs_array, list(val_accuracy), label="Validation", color="Green")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Training and Validation Accuracy on 50000 images")
    plt.legend()
    plt.show()
    np.save('/Users/shreyasvaderiyattil/Documents/W1_',myNet.W1)
    np.save('/Users/shreyasvaderiyattil/Documents/W2_',myNet.W2)
    np.save('/Users/shreyasvaderiyattil/Documents/b1_',myNet.b1)
    np.save('/Users/shreyasvaderiyattil/Documents/b2_',myNet.b2)
    '''
else:
    myNet.W1 = np.load('/Users/shreyasvaderiyattil/Documents/mlp2000weights/W1.npy')
    myNet.W2 = np.load('/Users/shreyasvaderiyattil/Documents/mlp2000weights/W2.npy')
    myNet.b1 = np.load('/Users/shreyasvaderiyattil/Documents/mlp2000weights/b1.npy')
    myNet.b2 = np.load('/Users/shreyasvaderiyattil/Documents/mlp2000weights/b2.npy')

confusion_matrix = np.zeros((10,10))
total = np.zeros(10)
correct_test = 0

for i in range(len(test_images_np)):
    y_hat_test = myNet.forward(test_images_np[i])
    if np.argmax(y_hat_test) == np.argmax(test_onehot[i]):
        correct_test += 1
    total[np.argmax(test_onehot[i])] += 1
    confusion_matrix[np.argmax(test_onehot[i]), np.argmax(y_hat_test)] += 1
            
print("Confusion Matrix: \n")
print(confusion_matrix/total*100)
print("Test Accuracy: "+str(correct_test / len(test_images_np)*100))

fig, ax = plt.subplots(8, 8, figsize=(7, 7))

for i in range(0,64):
    I = myNet.W1[i].reshape((28,28))

    ax[int(i/8), i%8].imshow(I,cmap='gray')
    ax[int(i/8), i%8].axis('off')
    
plt.show()

## Template for ConvNet Code
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.view(-1,1,28,28))))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_cnn(convnet,  num_images, n_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(convnet.parameters(), lr=learning_rate, momentum=0.9)

    train_accuracy = []
    val_accuracy = []

    for i in range(n_epochs):            
        running_loss = 0.0
        j = 0
        while j < num_images:
            optimizer.zero_grad()
            
            outputs = convnet(torch.from_numpy(train_images_np[j:j+256]).float())
            labels = torch.tensor(train_labels_np[j:j+256]).long()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            j = j + 256
        
        correct_train = 0
        correct_val = 0
        for i in range(num_images):
            outputs_train = convnet(torch.from_numpy(train_images_np[i]).float())
            if torch.argmax(outputs_train) == train_labels_np[i]:
                correct_train += 1

        for i in range(len(val_images_np)):
            outputs_val = convnet(torch.from_numpy(val_images_np[i]).float())
            if torch.argmax(outputs_val) == val_labels_np[i]:
                correct_val += 1

        train_accuracy.append(correct_train/num_images*100)
        val_accuracy.append(correct_val/len(val_images_np)*100)
        print("Loss: " + str(running_loss/((num_images/256)+1)))

        
    return train_accuracy, val_accuracy


convnet = ConvNet()
num_images = 2000
n_epochs = 100
learning_rate = 1e-3
epochs_array  = np.arange(1,101,1)

if input("Train or Load data (t or l): ") == "t":

    train_accuracy, val_accuracy = train_cnn(convnet, num_images, n_epochs, learning_rate)
    plt.plot(epochs_array, list(train_accuracy),label="Train", color="Red")
    plt.plot(epochs_array, list(val_accuracy), label="Validation", color="Green")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Training and Validation Accuracy on 2000 images for CNN")
    plt.legend()
    plt.show()

    num_images = 50000
    train_accuracy, val_accuracy = train_cnn(convnet, num_images, n_epochs, learning_rate)
    plt.plot(epochs_array, list(train_accuracy),label="Train", color="Red")
    plt.plot(epochs_array, list(val_accuracy), label="Validation", color="Green")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Training and Validation Accuracy on 50000 images for CNN")
    plt.legend()
    plt.show()
    
  
   # torch.save(convnet.state_dict(), 'cnn5000weights/50000CNN.pth')
else:
    convnet.load_state_dict(torch.load('cnn5000weights/50000CNN.pth'))

confusion_matrix = np.zeros((10,10))
total = np.zeros(10)
correct_test = 0

for i in range(len(test_images_np)):
    outputs_test = convnet(torch.from_numpy(test_images_np[i]).float())
    if torch.argmax(outputs_test) == test_labels_np[i]:
            correct_test += 1
    total[test_labels_np[i]] += 1
    confusion_matrix[test_labels_np[i], torch.argmax(outputs_test)] += 1
            
print("Confusion Matrix: \n")
print(confusion_matrix/total*100)
print("Test Accuracy: "+str(correct_test / len(test_images_np)*100))
