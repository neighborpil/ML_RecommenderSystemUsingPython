import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Add, Flatten
from tensorflow.keras.layers import Dense, Concatenate, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adamax

from sklearn.utils import shuffle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# read csv
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('./u.data', names=r_cols, sep='\t', encoding='latin-1')
ratings = ratings[['user_id', 'movie_id', 'rating']].astype(int)  # remove timestamp

print(ratings.head())

# split train test set
TRAIN_SIZE = 0.75
ratings = shuffle(ratings)
cutoff = int(TRAIN_SIZE * len(ratings))
ratings_train = ratings.iloc[:cutoff]
ratings_test = ratings.iloc[cutoff:]


# initialize variables
K = 200
mu = ratings_train.rating.mean()
M = ratings.user_id.max() + 1 # dataset is ordered. be careful when I use it
N = ratings.movie_id.max() + 1


def RMSE(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


# keras model
user = Input(shape=(1, ))  # user input, shape: (None, 1)
item = Input(shape=(1, ))  # item input, shape: (None, 1)
P_embedding = Embedding(M, K, embeddings_regularizer=l2())(user)  # (M, 1, K), shape: (None, 1, 200)
Q_embedding = Embedding(N, K, embeddings_regularizer=l2())(item)  # (N, 1, K), shape: (None, 1, 200)
user_bias = Embedding(M, 1, embeddings_regularizer=l2())(user)  # (M, 1, ), shape: (None, 1, 1)
item_bias = Embedding(N, 1, embeddings_regularizer=l2())(item)  # (N, 1, ), shape: (None, 1, 1)

# concatenate layers
P_embedding = Flatten()(P_embedding)
Q_embedding = Flatten()(Q_embedding)
user_bias = Flatten()(user_bias)
item_bias = Flatten()(item_bias)
R = Concatenate()([P_embedding, Q_embedding, user_bias, item_bias])

# Neural network
R = Dense(2048)(R)
R = Activation('relu')(R)
R = Dense(256)(R)
R = Activation('linear')(R)
R = Dense(1)(R)

# model setting
model = Model(inputs=[user, item], outputs=R)
model.compile(
    loss=RMSE,
    optimizer=Adamax(),  # SGD() could be used
    metrics=[RMSE]
)
model.summary()

# model fitting
result = model.fit(
    x=[ratings_train.user_id.values, ratings_train.movie_id.values],
    y=ratings_train.rating.values - mu,
    epochs=65,
    batch_size=512,
    validation_data=(
        [ratings_test.user_id.values, ratings_test.movie_id.values],
        ratings_test.rating.values - mu
    )
)

# plot RMSE
plt.plot(result.history['RMSE'], label='Train RMSE')
plt.plot(result.history['val_RMSE'], label='Test RMSE')
plt.legend()
plt.show()

# prediction
user_ids = ratings_test.user_id.values[0:6]
movie_ids = ratings_test.movie_id.values[0:6]
predictions = model.predict([user_ids, movie_ids]) + mu
print('Actuals: \n', ratings_test[0:6])
print()
print('Predictions: \n', predictions)


def RMSE2(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))


user_ids = ratings_test.user_id.values
movie_ids = ratings_test.movie_id.values
y_pred = model.predict([user_ids, movie_ids]) + mu
y_pred = np.ravel(y_pred, order='C') # make 2d array to 1d array
y_true = np.array(ratings_test.rating)

print(RMSE2(y_true, y_pred))
