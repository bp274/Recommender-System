#!/bin/python3

import pandas as pd
import numpy as np
from ast import literal_eval
import copy
import math
import collections
from collections import defaultdict
from scipy import linalg
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
#from surprise import Reader, Dataset, SVD, evaluate
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

def build_chart(genre, percentile=0.90):
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

Romance = build_chart('Romance').head(15)

links_small = pd.read_csv('links_small.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

movies_metadata = movies_metadata.drop([19730, 29503, 35587])
movies_metadata['id'] = movies_metadata['id'].astype('int')

similar_movies_metadata = movies_metadata[movies_metadata['id'].isin(links_small)]
similar_movies_metadata.shape

similar_movies_metadata['tagline'] = similar_movies_metadata['tagline'].fillna('')
similar_movies_metadata['description'] = similar_movies_metadata['overview'] + similar_movies_metadata['tagline']
similar_movies_metadata['description'] = similar_movies_metadata['description'].fillna('')

tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(similar_movies_metadata['description'])
tfidf_matrix.shape

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

similar_movies_metadata = similar_movies_metadata.reset_index()
titles = similar_movies_metadata['title']
indices = pd.Series(similar_movies_metadata.index, index=similar_movies_metadata['title'])

def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]

bestAnalogous = get_recommendations('Toy Story')

df = pd.read_csv('ratings_small.csv')

noOfUsers = df.userId.unique().shape[0]
movieDict = {}
movieDictRev = {}
k = 0
for row in df.itertuples():
    if row[2] not in movieDict:
        movieDict[row[2]] = k
        movieDictRev[k] = row[2]
        k += 1

df['movieId'] = df['movieId'].map(movieDict)

noOfMovies = df.movieId.unique().shape[0]


ratings = np.zeros((noOfUsers, noOfMovies))
for row in df.itertuples():
    ratings[row[1] - 1, row[2] - 1] = row[3]
    
def getMean(arr):
    value = 0
    count = 0
    for i in range(len(arr)):
        if arr[i] > 0:
            value += arr[i]
            count += 1
    return value / count

mean = [None for _ in range(noOfUsers)]
for i in range(noOfUsers):
    mean[i] = getMean(ratings[i])

newRatings = ratings.copy()
for i in range(noOfUsers):
    for j in range(noOfMovies):
        if newRatings[i][j] > 0:
            newRatings[i][j] -= mean[i]

def pearsonCorrelation(ratings, kind):
    if kind == 'user':    
        similarity = ratings.dot(ratings.T)
    elif kind == 'item':
        similarity = ratings.T.dot(ratings)
    norms = np.array([np.sqrt(np.diagonal(similarity))])
    return (similarity / norms / norms.T)
    
userSimilarity = pearsonCorrelation(newRatings, 'user')
itemSimilarity = pearsonCorrelation(newRatings, 'item')

val = 0
count = 0
for i in range(noOfUsers):
    for j in range(noOfMovies):
        if ratings[i][j] > 0:
            val += ratings[i][j]
            count += 1
mu = val / count

bx = np.zeros(noOfUsers)
for i in range(noOfUsers):
    bx[i] = mean[i] - mu

bi = np.zeros(noOfMovies)
val = 0
count = 0
for i in range(noOfMovies):
    for j in range(noOfUsers):
        if ratings[j][i] > 0:
            val += ratings[j][i]
            count += 1
    bi[i] = val / count

b = np.zeros((noOfUsers, noOfMovies))
for i in range(noOfUsers):
    for j in range(noOfMovies):
        b[i][j] = mu + bx[i] + bi[j]

def kNearestNeighbours(k, neighbours, i):
    knn = []
    for p, q in neighbours:
        if len(knn) == k:
            break
        if ratings[q][i] > 0:
            knn.append([p, q])
    return knn

index = [i for i in range(noOfUsers)]
arr = []
for x in range(25):
    arr.append([[p, q] for p, q in sorted(zip(userSimilarity[x], index), reverse = True)])

newRatings = np.zeros((noOfUsers, noOfMovies))
for x in range(25):
    for i in range(noOfMovies):
        num = 0
        den = 0
        for sxy, y in kNearestNeighbours(5, arr[x], i):
            num += sxy * ratings[y][i]
            den += sxy
        newRatings[x][i] = num / den   
    
index1 = [i for i in range(noOfMovies)]
arr1 = []
for i in range(100):
    arr1.append([[p, q] for p, q in sorted(zip(itemSimilarity[i], index1), reverse = True)])
    
def kNearestNeighbours1(k, neighbours, x):
    knn = []
    for p, q in neighbours:
        if len(knn) == k:
            break
        if ratings[x][q] > 0:
            knn.append([p, q])
    return knn

newRatings1 = np.zeros((noOfUsers, noOfMovies))
for i in range(100):
    for x in range(noOfUsers):
        num = 0
        den = 0
        for sij, j in kNearestNeighbours1(5, arr1[i], x):
            num += sij * (ratings[x][j] - b[x][j])
            den += sij
        newRatings1[x][i] = b[x][i] +  num / den


def userMSE(prediction, actual):
    num = 0
    den = 0
    for i in range(10):
        for j in range(noOfMovies):
            if actual[i][j] > 0:
                num += (actual[i][j] - prediction[i][j]) ** 2
                den += 1
    
    return (num / den) ** 0.5

print('User-based CF MSE: ' + str(userMSE(newRatings, ratings)))

def itemMSE(prediction, actual):
    num = 0
    den = 0
    for i in range(noOfUsers):
        for j in range(100):
            if actual[i][j] > 0:
                num += (actual[i][j] - prediction[i][j]) ** 2
                den += 1
    
    return (num / den) ** 0.5

print('Item-based CF MSE: ' + str(itemMSE(newRatings1, ratings)))
