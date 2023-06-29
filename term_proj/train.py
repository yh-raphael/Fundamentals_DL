from tkinter.messagebox import NO
import numpy as np   
import torch
from network import Network
from dataloader import *
import torch.optim as optim
import os
from sklearn.metrics import confusion_matrix

def train(model, nbatch_size, train_samples, optimizer, criterion, epoch, nlog_interval):

    model.train()                                                                                           # train 모드 킨다.
    train_loader = torch.utils.data.DataLoader(train_samples, batch_size=nbatch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nBatchCount = 0
    for data, target in train_loader:
        nBatchCount += 1
        data = data.float().to(device)
        target = target.to(device)

        optimizer.zero_grad()                                                                               # Adam optimizer.
        output = model(data)                                                                                # MLP

        loss = criterion(output, target)                                                                    # Cross-entropy loss
        loss.backward()
        optimizer.step()                                                                                    # Adam.
        if nBatchCount % nlog_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, nBatchCount * len(data), len(train_loader.dataset), 100. * nBatchCount / len(train_loader), loss.item()))

def test(model, dev_samples, nBatchSize):
    model.eval()                                                                                            # "나는 평가모드야" 켜기!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    true_y_list = []
    pred_y_list = []

    with torch.no_grad():
        test_loader = torch.utils.data.DataLoader(dev_samples, batch_size=nBatchSize, shuffle=True)

        for data, true_y in test_loader:
            data = data.float().to(device)
            true_y = true_y.to(device)                              # 답안: (0,1,2,3 중)
                
            output = model(data)                                    # 모델 아웃풋 뽑기.
            pred_y = torch.argmax(output, axis=1)                   # softmax 값이 가장 크게 나온 arg 값(0,1,2,3 중에서) 뽑기.

            pred_y_list.extend(pred_y.tolist())
            true_y_list.extend(true_y.tolist())

    train_accuracy =  accuracy_score(true_y_list, pred_y_list)      # 채점해서 점수 매기기.
    return train_accuracy

def getconfusionmetrix(model, dev_samples):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    true_y_list = []
    pred_y_list = []

    with torch.no_grad():
        test_loader = torch.utils.data.DataLoader(dev_samples, batch_size=nBatchSize, shuffle=True)

        for data, true_y in test_loader:
            data = data.float().to(device)
            true_y = true_y.to(device)           
                
            output = model(data)
            pred_y = torch.argmax(output, axis=1)

            pred_y_list.extend(pred_y.tolist())
            true_y_list.extend(true_y.tolist())

    return confusion_matrix(true_y_list, pred_y_list, labels=[0, 1, 2, 3])

def getACC(model, dev_samples):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    true_y_list = []
    pred_y_list = []

    with torch.no_grad():
        test_loader = torch.utils.data.DataLoader(dev_samples, batch_size=nBatchSize, shuffle=True)

        for data, true_y in test_loader:
            data = data.float().to(device)
            true_y = true_y.to(device)           
                
            output = model(data)
            pred_y = torch.argmax(output, axis=1)

            pred_y_list.extend(pred_y.tolist())
            true_y_list.extend(true_y.tolist())

    return accuracy_score(true_y_list, pred_y_list)


def accuracy_score(label_y, predict_y):
    
    correct_count = 0
    for index, value in enumerate(label_y):
        if predict_y[index] == value:
            correct_count += 1
    
    return correct_count / len(label_y)

# train 하이퍼 파라미터들
fLearningRate = 0.001
nEpoch = 30
nBatchSize = 128                    # 배치사이즈
nlog_interval = 5                   # 로그 찍는 interval

nInputSize = 132
nOutputSize = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Network(nInputSize, nOutputSize).to(device)
optimizer = optim.Adam(model.parameters(), lr=fLearningRate)
       
criterion = torch.nn.CrossEntropyLoss()

# If you want to use full Dataset, please pass None to csvpath
strDataFolderPath = os.path.join('sample_image_folder', 'skeleton_npy')
skeleton_samples = SkeletonDataset(strDataFolderPath) 

train_size = int(0.8 * len(skeleton_samples))
test_size = len(skeleton_samples) - train_size

train_set, val_set = torch.utils.data.random_split(skeleton_samples, [train_size, test_size])       # 8:2 split

fmaxAcc = 0

str_BestModelpath = ''
for epoch in range(1, nEpoch):
    train(model, nBatchSize, train_set, optimizer, criterion, epoch, nlog_interval)                 # train
    test_acc = test(model, val_set, nBatchSize)                                                     # validate
    print('Dev accuracy ', test_acc)
    if fmaxAcc < test_acc:                                                                          # 기존 highest보다 높으면, fmaxAcc로 업데이트.
        fMaxAcc = test_acc
        str_BestModelpath = f'./models/model_{test_acc}.pkl'
        torch.save(model.state_dict(),str_BestModelpath)                                            # 현재까지의 최고 성능 모델 저장!
        
testmodel = Network(nInputSize, nOutputSize).to(device)                                             # 132, 4
testmodel.load_state_dict(torch.load(str_BestModelpath))

print(getconfusionmetrix(testmodel, val_set))                                                       # confusion_matrix 뽑기.
print(getACC(testmodel, val_set))                                                                   # accuracy_score 뽑기.
    
    
    
