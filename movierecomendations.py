import streamlit as st
import pandas as pd
import numpy as np

movies_cols = ['MovieID', 'Title', 'Genres']
movies = pd.read_csv('ml-1m/movies.dat', sep='::', names=movies_cols, encoding='latin-1', engine='python')
rating_matrix = pd.read_csv('Rmat.csv')

S_adjusted = pd.read_csv('adjusted_similarity_matrix.csv', index_col=0)

movie_rating_counts = rating_matrix.astype(bool).sum(axis=0)
movie_avg_ratings = rating_matrix.mean(axis=0, skipna=True)
popularity_df = pd.DataFrame({
    "MovieID": rating_matrix.columns,
    "RatingCount": movie_rating_counts,
    "AverageRating": movie_avg_ratings
}).sort_values(by=["RatingCount", "AverageRating"], ascending=[False, False])

popularity_df['MovieID'] = popularity_df['MovieID'].str.lstrip('m').astype(str)
movies['MovieID'] = movies['MovieID'].astype(str)

top_100_popular_movies = pd.merge(popularity_df.head(100), movies, on='MovieID', how='left')

def myIBCF(newuser, similarity_matrix, rating_matrix, popularity_df, top_k=10):
    newuser = np.array(newuser)
    
    if similarity_matrix.shape[0] != rating_matrix.shape[1]:
        raise ValueError("Similarity matrix and rating matrix dimensions do not align.")
    
    predictions = np.full(len(newuser), np.nan)
    
    for i in range(len(newuser)):
        if np.isnan(newuser[i]):
            similarities = similarity_matrix.iloc[i, :]
            rated_movies_indices = np.where(~np.isnan(newuser))[0]
            valid_similarities = similarities.iloc[rated_movies_indices]
            rated_movie_ratings = newuser[rated_movies_indices]
            
            non_na_mask = ~valid_similarities.isna()
            valid_similarities = valid_similarities[non_na_mask]
            rated_movie_ratings = rated_movie_ratings[non_na_mask]
            
            weighted_sum = np.sum(valid_similarities * rated_movie_ratings)
            similarity_sum = np.sum(np.abs(valid_similarities))

            if similarity_sum > 0:
                predictions[i] = weighted_sum / similarity_sum
        
    movie_ids = rating_matrix.columns
    prediction_df = pd.DataFrame({
        "MovieID": movie_ids,
        "PredictedRating": predictions
    })
    
    already_rated_movies = set(np.where(~np.isnan(newuser))[0])
    prediction_df = prediction_df[~prediction_df.index.isin(already_rated_movies)]
    prediction_df = prediction_df.sort_values(
        by=["PredictedRating", "MovieID"],
        ascending=[False, True]
    )

    recommended_movies = prediction_df.head(top_k).dropna(subset=["PredictedRating"])
    
    if len(recommended_movies) < top_k:
        remaining_movies = [
            movie for movie in popularity_df["MovieID"]
            if movie not in already_rated_movies
        ]
        remaining_df = pd.DataFrame({
            "MovieID": remaining_movies[:top_k - len(recommended_movies)],
            "PredictedRating": [None] * (top_k - len(recommended_movies))
        })
        recommended_movies = pd.concat([recommended_movies, remaining_df])
    return recommended_movies


st.title("Movie Recommender System")

st.subheader("Rate These Movies")
movie_ratings = {}
for _, row in top_100_popular_movies.iterrows():
    rating = st.slider(f"{row['Title']} ({row['Genres']})", 0, 5, 0)
    movie_ratings[row['MovieID']] = rating if rating > 0 else np.nan

if st.button("Get Recommendations"):
    user_ratings_vector = [np.nan] * rating_matrix.shape[1]
    for movie_id, rating in movie_ratings.items():
        if not pd.isna(rating):
            user_index = int(movie_id) - 1
            user_ratings_vector[user_index] = rating

    recommendations = myIBCF(user_ratings_vector, S_adjusted, rating_matrix, popularity_df)

    st.subheader("Top 10 Recommended Movies")
    for _, row in recommendations.iterrows():
        movie_details = movies[movies['MovieID'] == row['MovieID']]
        if not movie_details.empty:
            st.text(movie_details.iloc[0]['Title'])
