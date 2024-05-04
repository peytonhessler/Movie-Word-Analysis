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


movies_df = movies_df.dropna(subset=['tagline'])

movies_df = movies_df.dropna(subset=['release_date'])

#print(movies_df.dtypes)

movies_df['release_date'] = pd.to_datetime(movies_df['release_date'])

movies_df['Decade'] = movies_df['release_date'].dt.year

movies_df['Decade'] = movies_df['Decade'].astype(str).str[:-1] + '0'

# Convert the column back to integers
movies_df['Decade'] = movies_df['Decade'].astype(int)

movies_df = movies_df[movies_df['Decade']>1930]

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
plt.savefig("Genre By Decade (Taglines)")
plt.show()
'''
End of Genre Count Graph
'''







tokenizer = TreebankWordTokenizer()
all_words = ' '.join(movies_df['tagline']).lower()
tagline_words = re.findall(r'\b\w+\b', all_words)
#tagline_words = tokenizer.tokenize(all_words)


tagline_words = [word for word in tagline_words if word not in string.punctuation]


stop_words = set(stopwords.words('english'))
tagline_words = [word for word in tagline_words if word not in stop_words]


word_freq = Counter(tagline_words)


top_words = word_freq.most_common(10)


words, frequencies = zip(*top_words)
plt.figure(figsize=(20, 20))
plt.bar(words, frequencies)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 10 Most Common Words in Taglines')
plt.xticks(rotation=45)
plt.savefig("Tagline Words Top 10")
plt.show()



# Look into digrams where I see combo of words as well. if Must See is right next to each other then 
word_freq_by_decade = {}
for decade in range(1940, 2010, 10):
    taglines_decade = movies_df[(movies_df['Decade'] == decade)]['tagline']
    #print(taglines_decade)
    all_words_decade = ' '.join(taglines_decade).lower()
    tagline_words_decade = re.findall(r'\b\w+\b', all_words_decade)
    #tagline_words_decade = word_tokenize(all_words_decade)
    tagline_words_decade = [word for word in tagline_words_decade if word not in string.punctuation]
    tagline_words_decade = [word for word in tagline_words_decade if word not in stop_words]
    word_freq = Counter(tagline_words_decade)
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

colors = plt.cm.tab20.colors  # default colormap with 20 colors
num_colors = len(words)
custom_cmap = ListedColormap(colors * (num_colors // len(colors)) + colors[:num_colors % len(colors)])

fig, ax = plt.subplots(figsize=(12,9))
bars = []
bottoms = [0] * len(years)
for i, word in enumerate(words):
    bar = ax.bar(years, word_counts_per_year[word], bottom=bottoms, label=word, color=custom_cmap(i))
    bars.append(bar)
    bottoms = [sum(value) for value in zip(bottoms, word_counts_per_year[word])]


for j, year in enumerate(years):
    x = year + 0.5
    for i, word in enumerate(words):
        value = word_counts_per_year[word][j]
        if value != 0:
            y = sum(word_counts_per_year[w][j] for w in words[:i]) + value / 2
            ax.text(x, y, f'{value:.1f}%', va='center', color='black')
            ax.plot([year + 0.05, x - 0.05], [y, y], color='black', linestyle='-', linewidth=0.5)


ax.set_ylabel('Word Frequency (%)')
ax.set_xlabel('Decade')
ax.set_title('Word frequency per Decade')
#ax.legend()

ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("Tagline Word Percentage Decades")
plt.show()



# I was trying to run a graph that would show similarity between tagles and rating.
# Are unique taglines signifying that the movie is better than generic taglines?
'''
movies_df = movies_df.head(1000)


taglines = movies_df['tagline']
vectorizer = TfidfVectorizer()
tagline_vectors = vectorizer.fit_transform(taglines)
similarity_matrix = cosine_similarity(tagline_vectors)


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
plt.ylabel('Tagline Similarity')
plt.title('Movies Graphed by Similarity and Rating')
#plt.legend()
#plt.show()
'''