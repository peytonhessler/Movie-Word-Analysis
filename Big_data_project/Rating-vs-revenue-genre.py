import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv('Data/Genre_Revenue_Rating.csv')
print(df)

# Scatter plot
plt.figure(figsize=(10, 6))

# Define colors for each genre
colors = {'Action': 'red', 'Adventure': 'yellow', 'Animation': 'pink', 'Comedy': 'blue', 'Crime': 'purple', 'Documentary': 'cyan',
          'Drama': 'green', 'Family': '#8BC1F7', 'Fantasy': '#BDE2B9', 'Foreign': '#A2D9D9', 'History': '#B2B0EA',
          'Horror': '#F9E0A2', 'Music': '#F4B678', 'Mystery': '#C9190B', 'Romance': '#6A6E73', 'Science Fiction': '#2C0000',
          'Thriller': '#C58C00', 'War': '#8F4700', 'Western': '#4CB140'}

# Plot each point with color coding based on genre
for genre in df['genre'].unique():
    print(genre)
    genre_df = df[df['genre'] == genre]
    plt.scatter(genre_df['vote_average'], genre_df['revenue'], color=colors[genre], label=genre)

# Set plot title and labels
plt.title('Rating vs Revenue')
plt.xlabel('Rating')
plt.ylabel('Revenue')
plt.ylim(0, 100000000)  # Set y-axis limits from 0 to 10000
plt.xlim(0.5, 10)   # Set x-axis limits from 0.5 to 10

# Add legend
plt.legend()

# Show plot
plt.grid(True)
plt.show()
