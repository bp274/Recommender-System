#Importing the Libraries
import pandas as pd
import numpy as np
import warnings; warnings.simplefilter('ignore')

#reading the rating csv file and storing the data in a dataframe
df = pd.read_csv('ratings_small.csv')

#splitting the data into test set and training set
train = df.iloc[:80000, :]
test = df.iloc[80000:, :]

#mapping the movieId to get them within a certain range
movie_dict = {}
movie_dict_rev = {}
k = 1
for row in df.itertuples():
    if row[2] not in movie_dict:
        movie_dict[row[2]] = k
        movie_dict_rev[k] = row[2]
        k += 1

#finding the numbers of users and movies in the training set
no_of_movies = train.movieId.unique().shape[0]
no_of_users = train.userId.unique().shape[0]

#changing the values
train['movieId'] = train['movieId'].map(movie_dict)

#creating the utility matrix
ratings = np.zeros((no_of_users, no_of_movies))
for row in train.itertuples():
    ratings[row[1] - 1, row[2] - 1] = row[3]

#finding the mean of an array
def get_mean(userId):
    value = sum(train[train["userId"] == userId]["rating"])
    count = len(train[train["userId"] == userId])
    return value / count

#finding the mean ratings for each user
#how much a user would rate a movie on average

mean = []
for userId in range(no_of_users):
    mean.append(get_mean(userId + 1))

#mean normalization of the ratings
normalized_ratings = ratings.copy()
for i in range(no_of_users):
    for j in range(no_of_movies):
        if normalized_ratings[i][j] > 0:
            normalized_ratings[i][j] -= mean[i]

#finding the pearson correlation coefficient
def pearson_correlation(ratings, kind):
    if kind == 'user':    
        similarity = ratings.dot(ratings.T)
    elif kind == 'item':
        similarity = ratings.T.dot(ratings)
    norms = np.array([np.sqrt(np.diagonal(similarity))])
    return (similarity / norms / norms.T)

# finding the similarity between items and users with pearson correlation
user_similarity = pearson_correlation(normalized_ratings, 'user')
item_similarity = pearson_correlation(normalized_ratings, 'item')

#finding the mean movie ratings
#the rating that any user would rate any movie on average
def get_mu():
    value = sum(train["rating"])
    count = len(train)
    return value / count

mu = get_mu()

#how much more or less a user rates a movie on average than global average movie rating
def get_bx():
    bx = np.zeros(no_of_users)
    for i in range(no_of_users):
        bx[i] = mean[i] - mu
    return bx

bx = get_bx()

#the average rating of each movie by all the users
def get_bi(ratings):
    bi = np.zeros(no_of_movies)
    val = 0
    count = 0
    for i in range(no_of_movies):
        for j in range(no_of_users):
            if ratings[j][i] > 0:
                val += ratings[j][i]
                count += 1
        bi[i] = val / count
    return bi

bi = get_bi(ratings)

#global baseline estimate for each user and every movie
#based on how much a user rates a movie on average and how much the movie is rated by an average user 
def get_b(mu, bx, bi):
    b = np.zeros((no_of_users, no_of_movies))
    for i in range(no_of_users):
        for j in range(no_of_movies):
            b[i][j] = mu + bx[i] + bi[j]
    return b

b = get_b(mu, bx, bi)

#finding the k-nearest neightbours
#the ratings the the k most similar users/movie have rated the movie
def k_nearest_neighbours(k, neighbours, i, kind):
    knn = []
    if kind == 'user':
        for p, q in neighbours:
            if ratings[q][i] > 0 and q != i:
                knn.append([p, q])
            if len(knn) == k:
                break
    elif kind == 'item':
        for p, q in neighbours:
            if ratings[x][q] > 0 and x != q:
                knn.append([p, q])
            if len(knn) == k:
                break
    return knn

#combining the user similarity with the user index
index = [i for i in range(no_of_users)]
arr = []
for x in range(no_of_users):
    arr.append([[p, q] for p, q in sorted(zip(user_similarity[x], index), reverse = True)])

#combining the item similarity with the items index
index1 = [i for i in range(no_of_movies)]
arr1 = []
for i in range(no_of_movies):
    arr1.append([[p, q] for p, q in sorted(zip(item_similarity[i], index1), reverse = True)])

#predicting the ratings based on similarity from other users
user_predicted_ratings = np.zeros((no_of_users, no_of_movies))
for x in range(10):
    for i in range(no_of_movies):
        num = 0
        den = 0
        for sxy, y in k_nearest_neighbours(5, arr[x], i, 'user'):
            num += sxy * ratings[y][i]
            den += sxy
        user_predicted_ratings[x][i] = num / den   

#predicting the ratings based on similarity from other items
item_predicted_ratings = np.zeros((no_of_users, no_of_movies))
for i in range(no_of_users):
    for x in range(no_of_users):
        num = 0
        den = 0
        for sij, j in k_nearest_neighbours(5, arr1[i], x, 'item'):
            num += sij * (ratings[x][j] - b[x][j])
            den += sij
        item_predicted_ratings[x][i] = b[x][i] +  num / den

"""
# User based prediction MSE
def user_mse(prediction, actual):
    num = 0
    den = 0
    for i in range(10):
        for j in range(no_of_movies):
            if actual[i][j] > 0:
                num += (actual[i][j] - prediction[i][j]) ** 2
                den += 1
    
    return (num / den) ** 0.5

print('User-based CF MSE: ' + str(user_mse(user_predicted_ratings, test_ratings)))

# Item bsed prediction MSE
def item_mse(prediction, actual):
    num = 0
    den = 0
    sum((prediction[i][j] - actual[i][j]) ** 2 for i in range(no_of_users) for j in range(no_of_movies) if actual[i][j] > 0)
    return (num / den) ** 0.5

print('Item-based CF MSE: ' + str(item_mse(item_predicted_ratings, test_ratings)))
"""
