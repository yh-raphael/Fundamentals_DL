## Troubleshooting Log

# 'data_utils.py' 파일의 line 6 "from scipy.misc import imread" 에서 scipy 가 없다고 에러 발생.  

아무리 이것저것 3시간 동안 설치를 시도해도 잘 안 됨.

$ conda create -n cs231n python=3.7  
이렇게 하면 된다더라..  


나는 python=3.8 로 하고 있었음.  

콘다 가상환경 하나 더 만들고 해보니..  
$ pip install -r requirements.txt  

귀신같이 됨..! scipy 깔리고 'data_utils.py' line 6 에서도 에러 안 남!  