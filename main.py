#!/bin/python3

import math
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.metrics import mean_squared_error
import warnings; warnings.simplefilter('ignore')


movies_metadata = pd.read_csv('movies_metadata.csv')

movies_metadata['genres'] = movies_metadata['genres'].fillna('[]')
movies_metadata['genres'] = movies_metadata['genres'].apply(literal_eval)
movies_metadata['genres'] = movies_metadata['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

i_c = movies_metadata['vote_count']
vote_counts = movies_metadata[i_c.notnull()]['vote_count'].astype('int')

i_a = movies_metadata['vote_average']
vote_averages = movies_metadata[i_a.notnull()]['vote_average'].astype('int')

C = vote_averages.mean()
m = vote_counts.quantile(0.95)

release_date = movies_metadata['release_date']
movies_metadata['year'] = pd.to_datetime(release_date, errors='coerce')
movies_metadata['year'] = movies_metadata['year'].apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

qualified_movies = movies_metadata[(i_c >= m) & (i_c.notnull()) & (i_a.notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified_movies['vote_count'] = qualified_movies['vote_count'].astype('int')
qualified_movies['vote_average'] = qualified_movies['vote_average'].astype('int')
qualified_movies.shape


def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

qualified_movies['wr'] = qualified_movies.apply(weighted_rating, axis=1)
qualified_movies = qualified_movies.sort_values('wr', ascending=False).head(250)
dataset = qualified_movies.head(15)


s = movies_metadata.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_md = movies_metadata.drop('genres', axis=1).join(s)

def build_chart(genre, percentile=0.85):
    df = gen_md[gen_md['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)

    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')

    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)

    return qualified

dataset2 = build_chart('Romance').head(15)
dataset3 = build_chart('Action').head(15)

df = pd.read_csv('ratings_small.csv')

noOfUsers = df.userId.unique().shape[0]
movieDict = {}
k = 0
for row in df.itertuples():
    if row[2] not in movieDict:
        movieDict[row[2]] = k
        k += 1

df['movieId'] = df['movieId'].map(movieDict)

noOfMovies = df.movieId.unique().shape[0]


ratings = np.zeros((noOfUsers, noOfMovies))
for row in df.itertuples():
    ratings[row[1] - 1, row[2] - 1] = row[3]

def similarity(ratings, kind, epsilon):
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
    elif kind == 'movie':
        sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)

userSimilarity = similarity(ratings, 'user', 10 ** (-9))
movieSimilarity = similarity(ratings, 'movie', 10 ** (-9))

def predict(ratings, similarity, kind):
    if kind == 'user':
        return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif kind == 'movie':
        return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])

def mse(prediction, actual):
    prediction = prediction[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(prediction, actual)

moviePrediction = predict(ratings, movieSimilarity, kind='movie')
userPrediction = predict(ratings, userSimilarity, kind='user')

print('User-based CF MSE: ' + str(mse(userPrediction, ratings)))
print('Item-based CF MSE: ' + str(mse(moviePrediction, ratings)))
