import streamlit as st
import pandas as pd
import numpy as np
import os

images_folder = './MovieImages'
movies_file_path = './ml-1m/movies.dat'
ratings_file_path = './Rmat.csv'
similarity_matrix_path = './adjusted_similarity_matrix.csv'

@st.cache_data
def load_movies():
    return pd.read_csv(
        movies_file_path,
        sep='::',
        engine='python',
        names=['MovieID', 'Title', 'Genres'],
        encoding='ISO-8859-1'
    )
@st.cache_data
def load_rating_matrix():
    return pd.read_csv(ratings_file_path)
@st.cache_data
def load_similarity_matrix():
    return pd.read_csv(similarity_matrix_path, index_col=0)
@st.cache_data
def compute_popularity_df(rating_matrix):
    ratings_filled = rating_matrix.fillna(0)
    movie_rating_counts = ratings_filled.astype(bool).sum(axis=0)
    movie_avg_ratings = rating_matrix.mean(axis=0, skipna=True)
    return pd.DataFrame({
        "MovieID": rating_matrix.columns,
        "RatingCount": movie_rating_counts,
        "AverageRating": movie_avg_ratings
    }).sort_values(by=["RatingCount", "AverageRating"], ascending=[False, False])
movies_df = load_movies()
rating_matrix = load_rating_matrix()

star_ratings = ["★☆☆☆☆", "★★☆☆☆", "★★★☆☆", "★★★★☆", "★★★★★"]
S_adjusted = load_similarity_matrix()
movie_rating_counts = rating_matrix.astype(bool).sum(axis=0)
popularity_df = compute_popularity_df(rating_matrix)

def get_image_path(movie_id):
    image_file = f"{movie_id}.jpg"
    image_path = os.path.join(images_folder, image_file)
    return image_path if os.path.exists(image_path) else None

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
#             print(f"Similarities for movie {i}: {valid_similarities}")
#             print(f"Weighted sum for movie {i}: {weighted_sum}")

            if similarity_sum > 0:
                predictions[i] = weighted_sum / similarity_sum
                #print(f"Movie {i}: Weighted Sum = {weighted_sum}, Similarity Sum = {similarity_sum}, Predicted Rating = {predictions[i]}")
        
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

st.subheader("Step 1: Rate Movies")
st.write("Rate as many movies as you can to improve recommendations.")

num_movies_to_display = 40
popularity_df = compute_popularity_df(rating_matrix)


popularity_df['MovieID'] = popularity_df['MovieID'].str.lstrip('m').astype(str)
movies_df['MovieID'] = movies_df['MovieID'].astype(str)
popular_movies = pd.merge(
    popularity_df.head(num_movies_to_display),
    movies_df,
    on="MovieID",
    how="left"
)
movies_to_display = popular_movies

user_ratings = {}

num_columns = 5
rows = [movies_to_display.iloc[i:i + num_columns] for i in range(0, len(movies_to_display), num_columns)]

for row in rows:
    cols = st.columns(num_columns)
    for col, (_, movie) in zip(cols, row.iterrows()):
        with col:
            image_path = get_image_path(movie['MovieID'])
            if image_path:
                st.image(image_path, width=150)
                st.markdown(f"<div style=\"text-align:center; font-size:9px;\">{movie['Title']}</div>", unsafe_allow_html=True)
            else:
                st.write(movie['Title'])

            st.markdown(
                f"<span style='font-size:9px; color:gray;'>{movie['Genres']}</span>",
                unsafe_allow_html=True
            )

            rating = st.radio(
                "Select a rating",
                options=[1, 2, 3, 4, 5],
                format_func=lambda x: star_ratings[x - 1],
                index=None,
                key=f"rating_{movie['MovieID']}"
            )
            user_ratings[movie['MovieID']] = rating if rating is not None and rating > 0 else np.nan

st.subheader("Step 2: Discover Movies You Might Like")
if st.button("Get Recommendations"):
    st.write("Generating recommendations...")

    user_ratings_vector = [np.nan] * rating_matrix.shape[1]
    for movie_id, rating in user_ratings.items():
        if not pd.isna(rating):
            try:
                user_index = int(movie_id) - 1
                if 0 <= user_index < len(user_ratings_vector): 
                    user_ratings_vector[user_index] = rating
                else:
                    print(f"MovieID {movie_id} is out of range and will be skipped.")
            except ValueError:
                print(f"Invalid MovieID {movie_id}. Skipping...")

    recommendations = myIBCF(user_ratings_vector, S_adjusted, rating_matrix, popularity_df)
    st.write("Here are the top 10 movies we recommend for you:")
    for _, row in recommendations.iterrows():
        movie_details = movies_df[movies_df['MovieID'] == row['MovieID'].lstrip('m')]
        if not movie_details.empty:
            title = movie_details.iloc[0]['Title']
            predicted_rating = row['PredictedRating']
            movie_id = row['MovieID']

            image_path = get_image_path(movie_id.lstrip('m'))
            
            col1, col2 = st.columns([1, 3])
            with col1:
                if image_path:
                    st.image(image_path, width=120)
                else:
                    st.write("📷 Image not available")
            with col2:
                st.write(f"🎬 **{title}**")
                st.write(f"⭐ Predicted Rating: **{predicted_rating:.2f}**")
