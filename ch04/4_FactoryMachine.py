import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle

r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('./u.data', names=r_cols, sep='\t', encoding='latin-1')
ratings = ratings[['user_id', 'movie_id', 'rating']].astype(int) # timestamp 제거

print('ratings')
print(ratings.head())

# user encoding
user_dict = {}
print('ratings count: ', len(ratings))
print('user set: ', set(ratings['user_id']))
print('user count: ', len(set(ratings['user_id'])))
for i in set(ratings['user_id']):
    user_dict[i] = len(user_dict)
print('user dict: ', user_dict)  # key: user_id, value: 0부터 시작 하는 index
n_user = len(user_dict)  # user count

# item encoding
item_dict = {}
start_point = n_user
for i in set(ratings['movie_id']):
    item_dict[i] = start_point + len(item_dict)  # user 뒤에 아이템 배치
n_item = len(item_dict)
start_point += n_item
num_x = start_point  # total number of x
ratings = shuffle(ratings, random_state=1)

# generate x data
data = []  # 변수 x의 값을 [인덱스, 값]의 형태로 저장할 변수
y = []  # 평점 데이터를 저장할 변수
w0 = np.mean(ratings['rating'])  # 전체 편향값
for i in range(len(ratings)):
    case = ratings.iloc[i]
    x_index = []  # 하나의 행에 대하여 변수 x의 인덱스를 기록
    x_value = []  # 변수값 기록
    if i < 3:
        print('case: ', case['user_id'])  # user_id
        print('user_dict: ', user_dict[case['user_id']])
    x_index.append(user_dict[case['user_id']])  # one hot encoding
    x_value.append(1)
    x_index.append(item_dict[case['movie_id']])  # one hot encoding
    x_value.append(1)
    data.append([x_index, x_value])
    y.append(case['rating'] - w0)  # 평점 - 전체평균(편향) 저장
    if (i % 10000) == 0:
        print('Encoding ', i, ' cases')

def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

class FM():
    '''
    Args:
        N: 변수 x의 수
        K: latent feature의 수
        data: [인덱스, 값]을 가지는 변수 x
        y: 평점
        alpha: learning rate
        beta: 정규화 계수
        train_ratio: 테스트셋의 비율
        iterations: 반복횟수
        tolerance: 반복을 중단하는 RMSE의 기준
        l2_reg: 정규화 여부
        verbose: 학습상황 보여줄지 여부
    '''
    def __init__(self, N, K, data, y, alpha, beta, train_ratio=0.75, iterations=100,
                 tolerance=0.005, l2_reg=True, verbose=True):
        self.K = K
        self.N = N
        self.n_cases = len(data)
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.l2_reg = l2_reg
        self.tolerance = tolerance
        self.verbose = verbose

        # initialize w




