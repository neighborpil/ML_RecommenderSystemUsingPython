# ML_RecommenderSystemUsingPython
hands on codes of book named recommender system using python in Korean

## Enviroment
 - create enviroment
 - 2022.10.7. 현재 시점에 python 3.10은 scikit-surprise패키지가 사용 불가
```
% conda create --name recommender python=3.8
```
 - ativate enviroment
```
% conda activate recommender
```
### packages
```
% conda install -y numpy pandas
% conda install -y jupyter notebook
% conda install -y scipy scikit-learn
% conda install -y seaborn matplotlib
% conda install -c conda-forge scikit-surprise

% conda install -c apple tensorflow-deps
% python -m pip install tensorflow-macos

% conda install selenium
% conda install -c transformer
```


## 전이학습 환경설정
```
% conda update -n base -c defaults conda
% conda create --name transfer python=3.8
% conda install -y numpy pandas jupyter notebook scipy scikit-learn seaborn matplotlib
# 맥용
% conda install -c apple tensorflow-deps
% python -m pip install tensorflow-macos
# 윈도우즈용
> python -m pip install tensorflow
% conda install -y tokenizers=0.12.1
% conda install -y transformers
% conda install -y pytorch
```
