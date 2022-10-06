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

print('data 0: ', data[0])  # [[user_index, item_index], [user_value, item_value]]

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

        # initialize w(w: 길이가 모든 변수의 수만큼인 배열
        self.w = np.random.normal(scale=1./self.N, size=(self.N))
        print('w.shape: ', self.w.shape)

        # initialize v (변수의 수 x 잠재변수의 수)
        self.v = np.random.normal(scale=1./self.K, size=(self.N, self.K))
        print('v.shape: ', self.v.shape)

        # split train/test
        cutoff = int(train_ratio * len(data))
        self.train_x = data[:cutoff]
        self.test_x = data[cutoff:]
        self.train_y = y[:cutoff]
        self.test_y = y[cutoff:]

    # training 하면서 rmse 계산
    def test(self):
        # sgd를 iteration만큼 수행
        best_RMSE = 10000
        best_iteration = 0
        training_process = []
        for i in range(self.iterations):
            rmse1 = self.sgd(self.train_x, self.train_y) # sgd & train rmse 계산
            rmse2 = self.test_rmse(self.test_x, self.test_y) # test rmse 계산
            training_process.append((i, rmse1, rmse2))
            if self.verbose:
                if (i+1) % 10 == 0:
                    print('Iteration: %d ; Train rmse = %.6f ; Test rmse = %.6f' % (i+1, rmse1, rmse2))
            if best_RMSE > rmse2:
                best_RMSE = rmse2
                best_iteration = i
            elif (rmse2 - best_RMSE) > self.tolerance: # rmse will increase over tolerance
                break

        print(best_iteration, best_RMSE)
        return training_process

    # w, v, 업데이트를 위한 sgd
    def sgd(self, x_data, y_data):
        y_pred = []
        for data, y in zip(x_data, y_data):
            x_idx = data[0]  # [[user_index, item_index], [user_value, item_value]]
            x_0 = np.array(data[1])  # xi
            x_1 = x_0.reshape(-1, 1)
            # print('x_idx: ', x_idx)  # [user_index, item_index]
            # print('x_0', x_0)  # [user_value, item_value]
            # print('x_0.shape', x_0.shape)
            # print('x_1', x_1)  # [[user_value], [item_value]]
            # print('x_1.shape', x_1.shape)

            # biases
            bias_score = np.sum(self.w[x_idx] * x_0)  # x_index에 해당하는 가중치 * x_value
            # print('bias_score: ', bias_score)

            # calculate score
            vx = self.v[x_idx] * (x_1)  # v matrix * x
            # print('v.shape', self.v.shape)  # (2625, 350)
            # print('v[x_idx]', self.v[x_idx])  # [[변수들]], 350개짜리 2차원 배열
            # print('v[x_idx].shape', self.v[x_idx].shape)  # (2, 350)
            # print('vx: ', vx)  # [[변수들]], 350개짜리 2차원 배열
            # print('vx.shape: ', vx.shape)  # (2, 350)
            sum_vx = np.sum(vx, axis=0)  # sigma(vx)
            # print('sum_vx: ', sum_vx)
            # print('sum_vx.shape: ', sum_vx.shape)  # (350, )
            sum_vx_2 = np.sum(vx * vx, axis=0)  # (v matrix * x) 의 제곱의 합
            latent_score = 0.5 * np.sum(np.square(sum_vx) - sum_vx_2)

            # 예측값 계산
            y_hat = bias_score + latent_score
            y_pred.append(y_hat)
            error = y - y_hat

            # w, v 업데이트
            if self.l2_reg:
                self.w[x_idx] += error * self.alpha * (x_0 - self.beta * self.w[x_idx])
                self.v[x_idx] += error * self.alpha * (x_1 * sum(vx) - (vx * x_1) - self.beta * self.v[x_idx])
            else:
                self.w[x_idx] += error * self.alpha * x_0
                self.v[x_idx] += error * self.alpha * (x_1 * sum(vx) - (vx * x_1))
        return RMSE(y_data, y_pred)

    def test_rmse(self, x_data, y_data):
        y_pred = []
        for data, y in zip(x_data, y_data):
            y_hat = self.predict(data[0], data[1])
            y_pred.append(y_hat)
        return RMSE(y_data, y_pred)

    def predict(self, idx, x):
        x_0 = np.array(x)
        x_1 = x_0.reshape(-1, 1)

        # biases
        bias_score = np.sum(self.w[idx] * x_0)

        # calculate score
        vx = self.v[idx] * x_1
        sum_vx = np.sum(vx, axis=0)
        sum_vx_2 = np.sum(vx * vx, axis=0)
        latent_score = 0.5 * np.sum(np.square(sum_vx) - sum_vx_2)

        # calculate prediction
        y_hat = bias_score + latent_score
        return y_hat

K = 350
fm1 = FM(num_x, K, data, y, alpha=0.0014, beta=0.075, train_ratio=0.75, iterations=600,
         tolerance=0.0005, l2_reg=True, verbose=True)
fm1.test()



