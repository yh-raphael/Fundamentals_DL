# Assignment 1
  
KNN, SVM, Softmax  

## Troubleshooting Log

### 'data_utils.py' 파일의 line 6 "from scipy.misc import imread" 에서 scipy 가 없다고 에러 발생.

아무리 이것저것 3시간 동안 설치를 시도해도 잘 안 됨.  

$ conda create -n cs231n python=3.7  
이렇게 하면 된다더라..  


나는 python=3.8 로 하고 있었음.  

콘다 가상환경 하나 더 만들고 해보니..  
$ pip install -r requirements.txt  

귀신같이 됨..! scipy 깔리고 'data_utils.py' line 6 에서도 에러 안 남!  


# Assignment 2

Two layer network, feature 프로그램 작성  

neural_net.py : TwoLayerNet class  
feature.ipynb  
two_layer_net.ipyb  

## 과제 코드로 돌려보기

feature.ipynb : 19626m 30.9s (327.1 시간)  
two_layer_net.ipyb : 18593m 40.6s (309.9 시간)  

이후에도 끝나지 않아서 중단 시킴..


# Assignment 3

Fully-connected Neural Network, Batch Normalization, Dropout  

fc_net.py : Fully-Connected neural network - forward pass, backward pass  
layers.py : Batch Normalization, Dropout  

### Test files

FullyConnectedNets.ipynb  
BatchNormalization.ipynb  
Dropout.ipynb  

## 과제 코드로 돌려보기

FullyConnectedNets.ipynb : 10일 정도 걸림..  
BatchNormalization.ipynb : 5일 정도  
Dropout.ipynb            : 5일 정도  


# Assignment 4

Pytorch, Tensorflow Tutorial with CIFAR-10  

PyTorch.ipynb    : 3-layer ConvNet 구성, CIFAR-10 dataset 이용해 모델 train, evaluate.  
TensorFlow.ipynb : 3-layer ConvNet 구성, CIFAR-10 dataset 이용해 모델 train, evaluate.  

