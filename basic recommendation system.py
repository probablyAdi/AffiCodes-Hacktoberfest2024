import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Sample dataset: User ratings for different movies
data = {
    'UserID': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
    'MovieID': [101, 102, 103, 101, 104, 102, 103, 104, 101, 104],
    'Rating': [5, 3, 4, 4, 5, 2, 5, 5, 4, 5]
}

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)

# Create a pivot table (user-item matrix) where rows represent users and columns represent movies
movie_ratings = df.pivot_table(index='UserID', columns='MovieID', values='Rating')

# Fill missing values with 0 (indicating no rating given)
movie_ratings = movie_ratings.fillna(0)

# Calculate the similarity between users based on their ratings using cosine similarity
user_similarity = cosine_similarity(movie_ratings)

# Create a DataFrame from the similarity matrix
user_similarity_df = pd.DataFrame(user_similarity, index=movie_ratings.index, columns=movie_ratings.index)

# Function to recommend movies for a given user based on similar users
def recommend_movies(user_id, num_recommendations=2):
    # Get similar users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]
    
    # Get the user's watched movies
    user_watched = movie_ratings.loc[user_id]
    
    # Recommend movies that similar users liked but the current user hasn't watched yet
    recommended_movies = pd.Series(dtype='float64')
    for other_user in similar_users:
        other_user_watched = movie_ratings.loc[other_user]
        recommendations = other_user_watched[user_watched == 0]  # Movies the user hasn't watched
        recommended_movies = recommended_movies.append(recommendations[recommendations > 0])
    
    # Sort by the highest rating and return the top N recommendations
    recommended_movies = recommended_movies.sort_values(ascending=False).head(num_recommendations)
    
    return recommended_movies.index.tolist()

# Test the recommendation system
user_id = 1  # Example user
recommended_movies = recommend_movies(user_id, 3)
print(f"Recommended movies for User {user_id}: {recommended_movies}")
