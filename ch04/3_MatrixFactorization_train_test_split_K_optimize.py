import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle

r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('./u.data', names=r_cols, sep='\t', encoding='latin-1')
ratings = ratings[['user_id', 'movie_id', 'rating']].astype(int) # timestamp 제거

print('ratings')
print(ratings.head())

TRAIN_SIZE = 0.75

ratings = shuffle(ratings, random_state=1)
print('\nratings shuffle')
print(ratings.head())

cutoff = int(TRAIN_SIZE * len(ratings))
ratings_train = ratings.iloc[:cutoff]
ratings_test = ratings.iloc[cutoff:]

# R = np.array(ratings)
# print('R')
# print(R.shape)
# print(R)
# print("[one_id, i]")
# for i, one_id in enumerate(ratings):
#     print(one_id, i)




# MF Class
class MF():
    def __init__(self, ratings, K, alpha, beta, iterations, verbose=True):
        self.R = np.array(ratings)  # index랑 column명이 달린 Dataframe에서 2차원배열 숫자만 뽑아낸다
        # print('\nratings shape')
        # print(ratings.shape)
        # print('\nratings')
        # print(ratings)

        # print('\nR shape')
        # print(self.R.shape)
        # print('\nR')
        # print(self.R)

        # user_id, item_id를 R의 index와 매핑하기 위한 dictionary생성
        item_id_index = []
        index_item_id = []
        for i, one_id in enumerate(ratings):  # enumerate는 인덱스와 아이템으로 나눠준다, dataframe을 돌리면 index를 iterate한다
            # print(i, one_id)
            item_id_index.append([one_id, i])
            index_item_id.append([i, one_id])
        # print('\nitem_id_index count')
        # print(len(item_id_index))
        # print('\nitem_id_index')
        # print(item_id_index)

        self.item_id_index = dict(item_id_index)  # item_id, id로 된 2차원 배열을 dictionary로 바꿔준다
        # print('\nself.item_id_index')
        # print(self.item_id_index)
        self.index_item_id = dict(index_item_id)
        # print('\nself.index_item_id')
        # print(self.index_item_id)


        user_id_index = []
        index_user_id = []
        for i, one_id in enumerate(ratings.T):
            user_id_index.append([one_id, i])
            index_user_id.append([i, one_id])
        self.user_id_index = dict(user_id_index)
        self.index_user_id = dict(index_user_id)

        self.num_users, self.num_items = np.shape(self.R)
        self.K = K  # latent variable
        self.alpha = alpha  # learning rate
        self.beta = beta  # bias
        self.iterations = iterations
        self.verbose = verbose  # show progress or not

    # Root Mean Squared Error(RMSE) 계산
    def rmse(self):
        xs, ys = self.R.nonzero()
        self.predictions = []
        self.errors = []

        for x, y, in zip(xs, ys):
            prediction = self.get_prediction(x, y)
            self.predictions.append(prediction)
            self.errors.append(self.R[x, y] - prediction)

        self.predictions = np.array(self.predictions)
        self.errors = np.array(self.errors)

        return np.sqrt(np.mean(self.errors ** 2))

    # rating prediction for user i and item j
    def get_prediction(self, i, j):
        prediction = self.b + self.b_u[i] + self.b_d[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    # sgd to get optimized P and Q matrix
    def sgd(self):
        for i, j, r in self.samples:
            prediction = self.get_prediction(i, j)
            e = (r - prediction)

            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_d[j] += self.alpha * (e - self.beta * self.b_d[j])

            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i, :])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j, :])

    def set_test(self, ratings_test):
        test_set = []
        for i in range(len(ratings_test)):
            # ratings_test: user_id, movie_id, rating
            x = self.user_id_index[ratings_test.iloc[i, 0]]
            y = self.item_id_index[ratings_test.iloc[i, 1]]
            z = ratings_test.iloc[i, 2]
            test_set.append([x, y, z])
            self.R[x, y] = 0  # remove test data from R to exclude test data from train data
        self.test_set = test_set
        return test_set

    def test_rmse(self):
        error = 0
        for one_set in self.test_set:
            predicted = self.get_prediction(one_set[0], one_set[1])
            error += pow(one_set[2] - predicted, 2)
        return np.sqrt(error/len(self.test_set))

    # training 하면서 test set의 정확도를 계산
    def test(self):  # mf모델을 sgd방식으로 훈련하는 핵심 함수
        # P 행렬을 임의의 변수로 채운다(유저의 수 x 잠재변수의 수)
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        # Q 행렬을 임의의 변수로 채운다(아이템의 수 x 잠재변수의 수)
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))
        self.b_u = np.zeros(self.num_users)  # 사용자의 평가경향
        self.b_d = np.zeros(self.num_items)  # 아이템의 평가경향
        self.b = np.mean(self.R[self.R.nonzero()])  # 전체평균
        rows, columns = self.R.nonzero()
        # train set에 대해서 (사용자, 아이템, 평점) 데이터를 구성
        self.samples = [(i, j, self.R[i, j]) for i, j in zip(rows, columns)]

        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()  # sgd 방법으로 P, Q, b_u, b_d를 업데이트
            rmse1 = self.rmse()  # train 셋의 rmse
            rmse2 = self.test_rmse()  # test 셋의 rmse
            training_process.append((i+1, rmse1, rmse2))
            if self.verbose:
                if (i+1) % 10 == 0:
                    print('Iteration: %d ; Train RMSE = %.4f ; Test RMSE = %.4f' % (i+1, rmse1, rmse2))

        return training_process

    def get_one_prediction(self, user_id, item_id):
        return self.get_prediction(self.user_id_index[user_id], self.item_id_index[item_id])

    def full_prediction(self):
        return self.b + self.b_u[:, np.newaxis] + self.b_d[np.newaxis, :] + self.P.dot(self.Q.T)


# 최적의 K값 찾기
results = []
index = []
for K in range(50, 261, 10):
    print('K = ', K)
    R_temp = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    mf = MF(R_temp, K=K, alpha=0.001, beta=0.02, iterations=300, verbose=True)
    test_set = mf.set_test(ratings_test)
    result = mf.test()
    index.append(K)
    result.append(result)

# 최적의 iteration값 찾기
summery = []
for i in range(len(results)):
    RMSE = []
    for result in results[i]:  # 각 K값의 iteration에 대하여 RMSE값만 추출하여 RMSE 변수에 저장
        RMSE.append(result[2])  # result[2] : test set의 rmse
    min = np.min(RMSE)  # RMSE 최소값
    j = RMSE.index(min)  # 최소값의 인덱스
    summery.append([index[i], j+1, RMSE[j]])  # K, iteration, RMSE

# 그래프 그리기
plt.plot(index, [x[2] for x in summery])  # x: K, y: RMSE
plt.ylim(0.89, 0.94)
plt.xlabel('K')
plt.ylabel('RMSE')
plt.show()



