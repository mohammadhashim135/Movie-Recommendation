import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

column_names = ['user_id', 'item_id', 'rating', 'timestamp'] 
tsv_path = 'file.tsv'
csv_path = 'Movie_Id_Titles.csv'

df = pd.read_csv(tsv_path, sep='\t', names=column_names)
movie_titles = pd.read_csv(csv_path)

data = pd.merge(df, movie_titles, on='item_id')

ratings = pd.DataFrame(data.groupby('title')['rating'].mean())
ratings['num of ratings'] = pd.DataFrame(data.groupby('title')['rating'].count())

sns.set_style('whitegrid')
plt.figure(figsize=(10, 4))
ratings['num of ratings'].hist(bins=70)
plt.xlabel('Number of Ratings')
plt.ylabel('Frequency')
plt.title('Histogram of Number of Ratings')
plt.show()

plt.figure(figsize=(10, 4))
ratings['rating'].hist(bins=70)
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title('Histogram of Ratings')
plt.show()


pivot_table = data.pivot_table(index='user_id', columns='title', values='rating').fillna(0)

item_similarity = pivot_table.corr(method='pearson', min_periods=100)

def recommend_movies(movie_name, rating_threshold=100, top_n=10):
    similar_movies = pd.DataFrame(item_similarity[movie_name])
    similar_movies = similar_movies.join(ratings['num of ratings'])
    similar_movies = similar_movies[similar_movies['num of ratings'] > rating_threshold]
    similar_movies = similar_movies.sort_values(by=movie_name, ascending=False)
    return similar_movies.head(top_n)


recommended_movies = recommend_movies('Star Wars (1977)')
print(recommended_movies)
