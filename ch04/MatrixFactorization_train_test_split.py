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
        self.R = np.array(ratings)
        print('\nratings shape')
        print(ratings.shape)
        print('\nratings')
        print(ratings)

        print('\nR shape')
        print(self.R.shape)
        print('\nR')
        print(self.R)

        # user_id, item_id를 R의 index와 매핑하기 위한 dictionary생성
        item_id_index = []
        index_item_id = []
        print('\ni one_id')
        for i, one_id in enumerate(ratings):
            print(i, one_id)
            item_id_index.append([one_id, i])
            index_item_id.append([i, one_id])
        print('\nitem_id_index count')
        print(len(item_id_index))
        print('\nitem_id_index')
        print(item_id_index)

        print('\nindex_item_id count')
        print(len(index_item_id))
        print('\nindex_item_id')
        print(index_item_id)
        self.item_id_index = dict(item_id_index)
        print('\nself.item_id_index')
        print(self.item_id_index)
        self.index_item_id = dict(index_item_id)
        print('\nself.index_item_id')
        print(self.index_item_id)


        user_id_index = []
        index_user_id = []
        for i, one_id in enumerate(ratings.T):
            user_id_index.append([one_id, i])
            index_user_id.append([i, one_id])
        self.user_id_index = dict(user_id_index)
        self.index_user_id = dict(index_user_id)

        self.num_users, self.num_items = np.shape(self.R)
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.verbose = verbose

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

    def train(self):
        # initialize user-feature and movie-feature matrix
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # initialize bias
        self.b_u = np.zeros(self.num_users)
        self.b_d = np.zeros(self.num_items)
        self.b = np.mean(self.R[self.R.nonzero()])

        # list of training samples
        rows, columns = self.R.nonzero()
        self.samples = [(i, j, self.R[i, j]) for i, j in zip(rows, columns)]

        # sgd for given number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            rmse = self.rmse()
            training_process.append((i+1, rmse))
            if self.verbose:
                if (i+1) % 10 == 0:
                    print('Iteration: %d ; Train RMSE = %.4f' % (i+1, rmse))

        return training_process

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


# 전체 데이터 사용 MF
R_temp = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

mf = MF(R_temp, K=30, alpha=0.001, beta=0.02, iterations=100, verbose=True)
'''
train_process = mf.train()
print(train_process)
'''