''' This model compares the desciptions and taglines of different movies, and provides recommendations that have the most similar plot desciptions'''

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# For dot product
from sklearn.metrics.pairwise import linear_kernel

# Import data from the clean file
df = pd.read_csv('data/metadata_clean.csv')

# Import original dataset
orig_df = pd.read_csv('data/movies_metadata.csv')

# Add the useful features into the cleaned dataframe
df['overview'], df['id'] = orig_df['overview'], orig_df['id']

# Print head of file
#print(df.head());

# Define a TF-IDF Vectorizer Object. Remove all English stopwords
tfidf = TfidfVectorizer(stop_words='english')

# Replace NaN with an empty string
df['overview'] = df['overview'].fillna('')

# Construct the required TF-IDF matrix by applying the fit_transform method on the overview feature
tfidf_matrix = tfidf.fit_transform(df['overview'])

# Output the shape of tfdf_matrix
#print(tfidf_matrix.shape)

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Construct a reverse mapping of indices aand movie titles, and drop duplicate titles, if any
indices = pd.Series(df.index, index=df['title']).drop_duplicates()


# Function that takes in movies title as input and gives recommendations
def content_recommender(title, cosine_sim=cosine_sim, df=df, indices=indices):
    # Obtain teh index of teh movie that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all movies with that movie and convert it into a list of tuples as described above
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the cosine similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of teh 10 most similar movies. Ignore the first movie.
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # return the top 10 most similar movies
    return df['title'].iloc[movie_indices]



# Get recommendations

print(content_recommender('The Lion King'))
