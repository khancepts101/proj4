import streamlit as st
import pandas as pd
import os

images_folder = './MovieImages'

movies_file_path = './ml-1m/movies.dat'
movies_df = pd.read_csv(
    movies_file_path,
    sep='::',
    engine='python',
    names=['MovieID', 'Title', 'Genres'],
    encoding='ISO-8859-1'
)

def get_image_path(movie_id):
    image_file = f"{movie_id}.jpg"
    image_path = os.path.join(images_folder, image_file)
    return image_path if os.path.exists(image_path) else None

st.title("Movie Recommender")
st.subheader("Step 1: Rate as many movies as possible")
st.write("Select a rating for each movie to improve recommendations.")

num_movies_to_display = 50
movies_to_display = movies_df.head(num_movies_to_display)

# Show movies
num_columns = 5 
rows = [movies_to_display.iloc[i:i + num_columns] for i in range(0, len(movies_to_display), num_columns)]


for row in rows:
    cols = st.columns(num_columns)
    for col, (_, movie) in zip(cols, row.iterrows()):
        with col:
            image_path = get_image_path(movie['MovieID'])
            if image_path:
                st.image(image_path, caption=movie['Title'], width=150)
            else:
                st.write(movie['Title'])  # Fallback if no image is available

            #Fix text size
            st.markdown(
                f"<span style='font-size:12px; color:gray;'>{movie['Genres']}</span>",
                unsafe_allow_html=True
            )
            
            st.slider(
                f"Rate the movie!",
                0, 5, 0,
                key=f"rating_{movie['MovieID']}"
            )

# Generate recommendations (TBD)
# st.subheader("Step 2: Discover movies you might like")
# if st.button("Click here to get your recommendations"):
#     st.write("Here are some movies you might like!")
