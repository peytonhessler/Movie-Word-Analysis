import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies_df = pd.read_csv('Data/movies_metadata.csv')

movies_df = movies_df.dropna(subset=['overview'])

print("Example: Gargoyle")
input_overview = input("Enter what things you are looking for in a movie: ")
vectorizer = TfidfVectorizer()
overview_vectors = vectorizer.fit_transform(movies_df['overview'])
input_overview_vector = vectorizer.transform([input_overview])
similarities = cosine_similarity(input_overview_vector, overview_vectors)

most_similar_index = similarities.argmax()
most_similar_movie = movies_df.iloc[most_similar_index]

print("Most similar movie:")
print("Title:", most_similar_movie['original_title'])
print("overview:", most_similar_movie['overview'])
print("Genre:", most_similar_movie['genres'])
