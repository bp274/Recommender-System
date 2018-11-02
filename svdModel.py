#!/bin/python3

#Importing various libraries
import math
import pandas as pd
import numpy as np

#reading the ratings_small csv file
df = pd.read_csv("ratings_small.csv")

# finding the values of 
movie_dict = {}
movie_dict_rev = {}
k = 1
for row in df.itertuples():
    if row[2] not in movie_dict:
        movie_dict[row[2]] = k
        movie_dict_rev[k] = row[2]
        k += 1

# getting the number of users and movies
no_of_movies = df.movieId.unique().shape[0]
no_of_users = df.userId.unique().shape[0]

#mapping the values of movieId to keep them in a certain range
df['movieId'] = df['movieId'].map(movie_dict)

# creating the utility matrix
ratings = np.zeros((no_of_users, no_of_movies))
for row in df.itertuples():
    ratings[row[1] - 1, row[2] - 1] = row[3]


train_ratings = np.zeros((no_of_users, no_of_movies))
x, y = ratings.nonzero()
count = 0
for i, j in zip(x, y):
    if count != 0:
        train_ratings[i, j] = ratings[i, j]
    else:
        train_ratings[i][j] = 0
    count = (count + 1) % 8
    
def train():
    # User and Item latent factor matrices
    P = np.random.normal(scale = 1. / K, size = (users, K))
    Q = np.random.normal(scale = 1. / K, size = (items, K))
    
    # global estimate based on how much a user rates a movie on average
    # and how much a movie is rated on average
    bu = np.zeros(users)
    bi = np.zeros(items)
    b = np.mean(U[np.where(U > 0)])
    
    # the index of the actual ratings
    samples = [(i, j, U[i, j]) for i in range(users) for j in range(items) if U[i, j] > 0]
    
    # using stochaistic gradient descent to find the root mean square error
    for i in range(no_of_iterations):
        np.random.shuffle(samples)
        stochaistic_gradient_descent()
        rmse = get_rmse()
        print("Iteration: %d ; error = %.4f" % (i + 1, rmse))

#finding the RMSE fromt the actual ratings available
def get_rmse():
    xs, ys = U.nonzero()
    pred = get_matrix()
    err = 0
    count = 0
    for x, y in zip(xs, ys):
        err += (U[x, y] - pred[x, y]) ** 2
        count += 1
    err = math.sqrt(err / count)
    return err

# standard stochaistic gradient algorithm
def stochaistic_gradient_descent(self):
    for i, j, r in samples:
        prediction = get_prediction(i, j)
        err = (r - prediction)
        
        bu[i] += alpha * (err - lmbda * bu[i])
        bi[j] += alpha * (err - lmbda * bi[j])
        
        P[i, :] += alpha * (err * Q[j, :] - lmbda * P[i, :])
        Q[j, :] += alpha * (err * P[i, :] - lmbda * Q[j, :])

# find the predicted ratings
def get_prediction(i, j):
    prediction = b + bu[i] + bi[j] + P[i, :].dot(Q[j, :].T)
    return prediction

# generating the predicted ratings matrix
def get_matrix():
    return b + bu[:, np.newaxis] + bi[np.newaxis:,] + P.dot(Q.T)


users, items = train_ratings.shape
alpha = 0.04
no_of_iterations = 50
K = 100
lmbda = 0.1
train()
ans = get_matrix()

err = 0
num = 0
count = 0
for i, j in zip(x, y):
    if count == 0:
        err += (ans[i, j] - ratings[i, j]) ** 2
        num += 1
    count = (count + 1) % 8

print("Error :-", math.sqrt(err / num))