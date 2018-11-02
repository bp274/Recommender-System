#Importing Necessary Libraries
from __future__ import division, print_function
import numpy as np
import pandas as pd

from keras.layers import Input, Embedding, merge, Flatten, Dense, Dropout
from keras.regularizers import l2
from keras.models import Model
from keras.optimizers import Adam

#Importing the dataframe
ratings = pd.read_csv('ratings_small.csv')
ratings.head()
pd.read_csv('movies.csv')[:5]

movie_names = pd.read_csv('movies.csv').set_index('movieId')['title'].to_dict()

users  = ratings.userId.unique()
movies = ratings.movieId.unique()

numOfUsers  = ratings.userId.nunique()
numOfMovies = ratings.movieId.nunique()

#Scaling User Ids and Movies Ids  
useridtoidx = {o:i for i,o in enumerate (users)}
movieidtoidx= {o:i for i,o in enumerate (movies)}

#Fitting the Scaled Values
ratings.userId = ratings.userId.apply(lambda x:useridtoidx[x])
ratings.movieId = ratings.movieId.apply(lambda x:movieidtoidx[x])

user_min = ratings.userId.min()
user_max = ratings.userId.max()
movie_min = ratings.movieId.min()
movie_max = ratings.movieId.max()

lf_num = 50 #number of latent factors

#create an array of random booleans of same number of ratings
msk = np.random.rand(len(ratings))<0.8 
#Splitting Data into test set and training set
trn = ratings[msk]
val= ratings[~msk]

def create_embedding(in_num, out_num, input_shape, name, inp=None):
    if (inp == None):
        inp = Input(shape=(1,), dtype='int64', name=name)
    return inp, Embedding(in_num, out_num, input_length=input_shape, W_regularizer=l2(1e-4))(inp)

ui, ue = create_embedding(numOfUsers, lf_num, 1, 'ue')
mi, me = create_embedding(numOfMovies, lf_num,1, 'me')

#create biases
ubi, ub = create_embedding(numOfUsers, out_num=1, input_shape=1, name='ub', inp = ui)
umi, mb = create_embedding(numOfMovies, out_num=1, input_shape=1, name='mb', inp = mi)

ub = Flatten()(ub)
mb = Flatten()(mb)

#Building a simple model and calculating mean squared error 
x = merge([ue,me], mode='dot')
x = Flatten()(x)
x = merge ([x, ub], mode='sum')
x = merge ([x, mb], mode='sum')

model = Model([ui, mi], x)
model.compile(Adam(0.001), loss='mse')

#Function for training the model
def train(nb_epoch = 1, lr=0.001):
    model.optimizer.lr= lr
    model.fit([trn.userId, trn.movieId], trn.rating, nb_epoch = nb_epoch, 
          validation_data=([val.userId, val.movieId], val.rating), batch_size = 1000)

#Training and Saving the Simple Model model.
train(nb_epoch=3, lr=0.001)
train(nb_epoch=10, lr=0.1)
train(nb_epoch=5 , lr=0.001)
model.save_weights('movielens_recommender_sys1_googlecolab.h5')

#Builing The ANN
x = merge([ue,me], mode='concat')
x = Flatten()(x)
x = Dropout(0.3)(x)
#Adding first layer with 100 neurons
x = Dense(100, activation='sigmoid')(x)
x = Dropout(0.5)(x)
#Adding anothe layer with 50 neurons
x = Dense(50, activation='sigmoid')(x)
x = Dropout(0.5)(x)
#Adding Another Layer
x = Dense(25, activation='sigmoid')(x)
x = Dropout(0.7)(x)
#Adding the last layer
x = Dense(1 )(x)
model = Model([ui, mi], x)
#Compiling the Model and Using Adam as Optimizer
model.compile(Adam(0.001), loss='mse')

# Training the model
train(nb_epoch=5 , lr = 0.01)
train(nb_epoch=10, lr=0.1)
train(nb_epoch=10, lr=0.001)
train(nb_epoch=10, lr=0.0001)
model.save_weights('movielens_recommender_sys_googlecolab.h5')

model.load_weights('movielens_recommender_sys_googlecolab.h5')

user_id = 90; movie_id = 5
p = model.predict([np.array([user_id]), np.array([movie_id])])
print ('User %d would likely rate movie \'%s\' at %f'%(user_id, movie_names[movies[movie_id]], p))

#Building Utility Matrix using predictions from Neural Network
predictedRatings = np.zeros((numOfUsers,numOfMovies))
for i in range(numOfUsers):
  for j in range(numOfMovies):
    predictedRatings[i][j] = model.predict([np.array([i]), np.array([j])])
    
#Saving the Utility Matrix in a csv file.    
df = pd.DataFrame(predictedRatings, columns=[str(x) for x in range(numOfMovies)])
df.dtypes
df.to_csv('PredictedRatingsNN.csv')
df.head()

df = pd.read_csv('PredictedRatingsNN.csv')

g = ratings.groupby('movieId')['rating'].count()
topMovies = g.sort_values(ascending=False)[:2000] #get the top 2000 so that it is easier for us to analyze
topMovies = np.array(topMovies.index)

get_movie_bias = Model(mi, mb) # a simple model that takes movie index and returns the movie bias
movie_bias = get_movie_bias.predict(topMovies) #get the biases
movie_ratings = [(b[0], movie_names[movies[i]]) for i,b in zip(topMovies,movie_bias)] #combine the movie bias and movie name

sorted(movie_ratings)[:5] #-> lowest rated movies