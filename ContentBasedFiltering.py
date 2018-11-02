#Importing All the Neccessary Libraries

import pandas as pd
import numpy as np
from ast import literal_eval
from scipy import linalg
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from surprise import Reader, Dataset, SVD, evaluate
import warnings; warnings.simplefilter('ignore')


# Part 1 :- CONTENT BASED FILTERING
# Based on average ratings assigned to each movie
# and on different factors like cast and crew

#Importing the dataset
movies_metadata = pd.read_csv('movies_metadata.csv')

#Preprocessing the given Dataset on the basis of Genre
movies_metadata['genres'] = movies_metadata['genres'].fillna('[]')
movies_metadata['genres'] = movies_metadata['genres'].apply(literal_eval)
movies_metadata['genres'] = movies_metadata['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

#Preprocessing Vote Count
i_c = movies_metadata['vote_count']
vote_counts = movies_metadata[i_c.notnull()]['vote_count'].astype('int')
#Preprocessing Vote Average
i_a = movies_metadata['vote_average']
vote_averages = movies_metadata[i_a.notnull()]['vote_average'].astype('int')

#Calclating Vote Averages
C = vote_averages.mean()
m = vote_counts.quantile(0.95)

#Processing Year of releasing
release_date = movies_metadata['release_date']
movies_metadata['year'] = pd.to_datetime(release_date, errors='coerce')
movies_metadata['year'] = movies_metadata['year'].apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

#Finding Movies Having Votes more than Average
qualified_movies = movies_metadata[(i_c >= m) & (i_c.notnull()) & (i_a.notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]
qualified_movies['vote_count'] = qualified_movies['vote_count'].astype('int')
qualified_movies['vote_average'] = qualified_movies['vote_average'].astype('int')
qualified_movies.shape

#Function for calculating weigted ratings
def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

#Defining Weighted Ratings on basis of vote counts and averages
qualified_movies['wr'] = qualified_movies.apply(weighted_rating, axis=1)
qualified_movies = qualified_movies.sort_values('wr', ascending=False).head(250)
dataset = qualified_movies.head(15)


s = movies_metadata.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_md = movies_metadata.drop('genres', axis=1).join(s)

#Function For finding top Movies in a particular genre
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

#Processing Id of Movies
movies_metadata = movies_metadata.drop([19730, 29503, 35587])
movies_metadata['id'] = movies_metadata['id'].astype('int')

similar_movies_metadata = movies_metadata[movies_metadata['id'].isin(links_small)]
similar_movies_metadata.shape

#Processing Description of Movies
similar_movies_metadata['tagline'] = similar_movies_metadata['tagline'].fillna('')
similar_movies_metadata['description'] = similar_movies_metadata['overview'] + similar_movies_metadata['tagline']
similar_movies_metadata['description'] = similar_movies_metadata['description'].fillna('')

#Encoding Description of Movies
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(similar_movies_metadata['description'])
tfidf_matrix.shape

#Finding Cosine Similarity Using LinearKernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

similar_movies_metadata = similar_movies_metadata.reset_index()
titles = similar_movies_metadata['title']
indices = pd.Series(similar_movies_metadata.index, index=similar_movies_metadata['title'])

#Recommending On the basis of Similarity Scores
def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]

bestAnalogous = get_recommendations('The Dark Knight')

#Reading the dataframe
credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')

#Preprocessing the data for better prediction on the basis of cast,crew,director,keywords etc.
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
movies_metadata['id'] = movies_metadata['id'].astype('int')
movies_metadata.shape
#Merging movies_metadata ,credits ,keywords on id
movies_metadata = movies_metadata.merge(credits, on='id')
movies_metadata = movies_metadata.merge(keywords, on='id')

similar_movies_metadata = movies_metadata[movies_metadata['id'].isin(links_small)]
similar_movies_metadata.shape

similar_movies_metadata['cast'] = similar_movies_metadata['cast'].apply(literal_eval)
similar_movies_metadata['crew'] = similar_movies_metadata['crew'].apply(literal_eval)
similar_movies_metadata['keywords'] = similar_movies_metadata['keywords'].apply(literal_eval)
similar_movies_metadata['cast_size'] = similar_movies_metadata['cast'].apply(lambda x: len(x))
similar_movies_metadata['crew_size'] = similar_movies_metadata['crew'].apply(lambda x: len(x))

#Function for finding director of movie
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

#Processing cast, crew, director etc.
similar_movies_metadata['director'] = similar_movies_metadata['crew'].apply(get_director)
similar_movies_metadata['cast'] = similar_movies_metadata['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
similar_movies_metadata['cast'] = similar_movies_metadata['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)
similar_movies_metadata['keywords'] = similar_movies_metadata['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
similar_movies_metadata['cast'] = similar_movies_metadata['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
similar_movies_metadata['director'] = similar_movies_metadata['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
similar_movies_metadata['director'] = similar_movies_metadata['director'].apply(lambda x: [x,x, x])

s = similar_movies_metadata.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'

s = s.value_counts()
s[:5]
s = s[s > 1]
#Using SnowballStemmer for plural to singular conversion
stemmer = SnowballStemmer('english')
stemmer.stem('dogs')

def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words

similar_movies_metadata['keywords'] = similar_movies_metadata['keywords'].apply(filter_keywords)
similar_movies_metadata['keywords'] = similar_movies_metadata['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
similar_movies_metadata['keywords'] = similar_movies_metadata['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

similar_movies_metadata['soup'] = similar_movies_metadata['keywords'] + similar_movies_metadata['cast'] + similar_movies_metadata['director'] + similar_movies_metadata['genres']
similar_movies_metadata['soup'] = similar_movies_metadata['soup'].apply(lambda x: ' '.join(x))

#Encoding of Directors,Genre,Keywords etc.
count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(similar_movies_metadata['soup'])

#Again finding Cosine Similarity
cosine_sim = cosine_similarity(count_matrix, count_matrix)

similar_movies_metadata = similar_movies_metadata.reset_index()
titles = similar_movies_metadata['title']
indices = pd.Series(similar_movies_metadata.index, index=similar_movies_metadata['title'])

#Function for recommending on basis of new Similarity Scores
def improved_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]

    movies = similar_movies_metadata.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    return qualified

analogous = improved_recommendations('The Dark Knight')
