import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from collections import Counter
import nltk
#nltk.download('punkt')
#from nltk.tokenize import word_tokenize
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
#nltk.download('stopwords')
import string
import re
from matplotlib.colors import ListedColormap
import ast


movies_df = pd.read_csv('Data/movies_metadata.csv')


movies_df = movies_df.dropna(subset=['overview'])

movies_df = movies_df.dropna(subset=['release_date'])

#print(movies_df.dtypes)

movies_df['release_date'] = pd.to_datetime(movies_df['release_date'])

movies_df['Decade'] = movies_df['release_date'].dt.year

movies_df['Decade'] = movies_df['Decade'].astype(str).str[:-1] + '0'

# Convert the column back to integers
movies_df['Decade'] = movies_df['Decade'].astype(int)

movies_df = movies_df[(movies_df['Decade']>1930) & (movies_df['Decade']<2020)]

# Display the DataFrame with the new "Decade" column
#print(movies_df['Decade'])


'''
Start of Genre Count Graph
'''

movies_df['genres'] = movies_df['genres'].apply(ast.literal_eval)


def count_genres_in_group(group):
    flattened_genres = [genre['name'] for sublist in group['genres'] for genre in sublist]
    return Counter(flattened_genres)


genre_counts_by_decade = movies_df.groupby('Decade').apply(count_genres_in_group)
print(genre_counts_by_decade)

genre_counts_df = pd.DataFrame(genre_counts_by_decade.tolist(), index=genre_counts_by_decade.index).fillna(0)
print(genre_counts_df)


genre_counts_df['Total'] = genre_counts_df.sum(axis=1)
print("genre-counts")
print(genre_counts_df)

total_movies_per_decade = genre_counts_df['Total'].sum()

x=0
for genre in genre_counts_df.columns[:-1]:
    print(genre_counts_df[genre])
    # I figured out how to count everything, but I can't get rid of the percentage text without it breaking?
    genre_counts_df[genre + '_Percentage'] = (genre_counts_df[genre] / genre_counts_df['Total']) * 100

df = genre_counts_df.iloc[:, 21:]
print("DF")
print(df)
#print(data)

print("Testing genre count percentage")
print(genre_counts_df)


# Plotting
df.plot(kind='bar', stacked=True, figsize=(20,20))
plt.xlabel('Decade')
plt.ylabel('Number of Genres')
plt.title('Number of Genres per Decade')
plt.xticks(rotation=0)
plt.legend(title='Genre', loc='upper left', bbox_to_anchor=(1, 1))
#plt.show()
plt.savefig("Genre By Decade (Overviews)")
plt.show()
'''
End of Genre Count Graph
'''







tokenizer = TreebankWordTokenizer()
all_words = ' '.join(movies_df['overview']).lower()
overview_words = re.findall(r'\b\w+\b', all_words)
#overview_words = tokenizer.tokenize(all_words)


overview_words = [word for word in overview_words if word not in string.punctuation]


stop_words = set(stopwords.words('english'))
overview_words = [word for word in overview_words if word not in stop_words]


word_freq = Counter(overview_words)


top_words = word_freq.most_common(10)


words, frequencies = zip(*top_words)
plt.figure(figsize=(20, 20))
plt.bar(words, frequencies)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Most Common Words in Overviews')
plt.xticks(rotation=45)
plt.savefig("Overview Words Top 10")
plt.show()



# Look into digrams where I see combo of words as well. if Must See is right next to each other then 
word_freq_by_decade = {}
for decade in range(1940, 2020, 10):
    overviews_decade = movies_df[(movies_df['Decade'] == decade)]['overview']
    #print(overviews_decade)
    all_words_decade = ' '.join(overviews_decade).lower()
    overview_words_decade = re.findall(r'\b\w+\b', all_words_decade)
    #overview_words_decade = word_tokenize(all_words_decade)
    overview_words_decade = [word for word in overview_words_decade if word not in string.punctuation]
    overview_words_decade = [word for word in overview_words_decade if word not in stop_words]
    word_freq = Counter(overview_words_decade)
    top_words = word_freq.most_common(10)
    word_freq_by_decade[decade] = top_words

print(word_freq_by_decade)


years = list(word_freq_by_decade.keys())
words = list({word for year in word_freq_by_decade.values() for word, _ in year})
word_counts_per_year = {word: [0] * len(years) for word in words}


for i, year in enumerate(years):
    total_count = sum(count for _, count in word_freq_by_decade[year])
    for word, count in word_freq_by_decade[year]:
        word_counts_per_year[word][i] = (count / total_count) * 100 if total_count != 0 else 0

# Assigning colors to words
colors = plt.cm.tab20.colors  # default colormap with 20 colors
num_colors = len(words)
custom_cmap = ListedColormap(colors * (num_colors // len(colors)) + colors[:num_colors % len(colors)])

# Plotting stacked bar chart
fig, ax = plt.subplots(figsize=(12,9))
bars = []
bottoms = [0] * len(years)
for i, word in enumerate(words):
    bar = ax.bar(years, word_counts_per_year[word], bottom=bottoms, label=word, color=custom_cmap(i))
    bars.append(bar)
    bottoms = [sum(value) for value in zip(bottoms, word_counts_per_year[word])]


for j, year in enumerate(years):
    x = year + 0.5  # Offset from the bar
    for i, word in enumerate(words):
        value = word_counts_per_year[word][j]
        if value != 0:
            y = sum(word_counts_per_year[w][j] for w in words[:i]) + value / 2
            ax.text(x, y, f'{value:.1f}%', va='center', color='black')
            ax.plot([year + 0.05, x - 0.05], [y, y], color='black', linestyle='-', linewidth=0.5)


ax.set_ylabel('Word Frequency (%)')
ax.set_xlabel('Decade')
ax.set_title('Overview Word frequency per Decade')
#ax.legend()

# Move the legend outside the graph
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Overview Word Percentage Decades")
plt.show()



# I was trying to run a graph that would show similarity between summary and rating.
# Are unique overviews signifying that the movie is better than generic overviews?
'''
movies_df = movies_df.head(1000)


overviews = movies_df['overview']
vectorizer = TfidfVectorizer()
overview_vectors = vectorizer.fit_transform(overviews)
similarity_matrix = cosine_similarity(overview_vectors)


max_similarity = 0
most_similar_movies = None
for i in range(len(similarity_matrix)):
    for j in range(i + 1, len(similarity_matrix)):
        similarity = similarity_matrix[i, j]
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_movies = (i, j)


plt.figure(figsize=(10, 6))
for i, movie in enumerate(movies_df.itertuples()):
    plt.scatter(movie.vote_average, similarity_matrix[i].mean(), label=movie.original_title)

plt.scatter([movies_df.iloc[most_similar_movies[0]].vote_average, movies_df.iloc[most_similar_movies[1]].vote_average],
            [similarity_matrix[most_similar_movies[0]].mean(), similarity_matrix[most_similar_movies[1]].mean()],
            color='red', label=f'Most Similar: {movies_df.iloc[most_similar_movies[0]].original_title} - {movies_df.iloc[most_similar_movies[1]].original_title}')

plt.xlabel('Rating')
plt.ylabel('overview Similarity')
plt.title('Movies Graphed by Similarity and Rating')
#plt.legend()
#plt.show()
'''