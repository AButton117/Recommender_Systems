''' This model uses the genre, director, movie's three major stars and sub genres
 or keywords to give a recommendation.'''

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# For dot product
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval

# Import data from the clean file
df = pd.read_csv('data/metadata_clean.csv')

# Import original dataset
orig_df = pd.read_csv('data/movies_metadata.csv')

#Load the keywords and credits files
cred_df = pd.read_csv('data/credits.csv')
key_df = pd.read_csv('data/keywords.csv')

# Add the useful features into the cleaned dataframe
df['overview'], df['id'] = orig_df['overview'], orig_df['id']



#print(cred_df.head())
#print(key_df.head())



#Convert bad data to NaN
def clean_ids(x):
    try:
        return int(x)
    except:
        return np.nan

#Clean the ids of df
df['id'] = df['id'].apply(clean_ids)

#Filter all rows that have a null ID
df = df[df['id'].notnull()]

#Convert IDs into integers
df['id'] = df['id'].astype('int')
key_df['id'] = key_df['id'].astype('int')
cred_df['id'] = cred_df['id'].astype('int')

#Merge keywords and credits into your main metadata dataframe
df = df.merge(cred_df, on='id')
df = df.merge(key_df, on='id')

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
#Convert the strified objects into the native pythonobjects from ast import literal_eval


features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df[feature] = df[feature].apply(literal_eval)

#Extract the director's name. If director is not listed, return NaN
    def get_director(x):
        for crew_member in x:
            if crew_member['job'] == 'Director':
                return crew_member['name']
        return np.nan
#Define the new director feture
df['director'] = df['crew'].apply(get_director)

#Returns the list top 3 elements or entire list, whichever is more
def generate_list(x):
    if isinstance(x, list):
        names = [ele['name'] for ele in x]
        #Check if more than 3 elements exist
        if len(names) > 3:
            names = names[:3]
            return names
    return []

#Apply the generate_list function to cast and keywords
df['cast'] = df['cast'].apply(generate_list)
df['keywords'] = df['keywords'].apply(generate_list)

#Only consider a maximum of 3 genres
df['genres'] = df['genres'].apply(lambda x: x[:3])

#Print the new feters of the 5 movies along with title
#print(df[['title', 'cast', 'director', 'keywords', 'genres']].head(3))

#Function to sanitize data to prevent ambiguity
#Removes spaces and converts to lowercase
def sanitize(x):
    if isinstance(x, list):
        #Strip spaces and convert to lowercase
        return [str.lower(i.replace(" "," ")) for i in x]
    else:
        #Check if director exists. If not, return to emtpy string
        if isinstance(x, str):
            return str.lower(x.replace(" "," "))
        else:
            return ''
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

#Apply the generate_list function to cast, keywords, director and genres
for feature in ['cast', 'director', 'genres', 'keywords']:
    df[feature] = df[feature].apply(sanitize)

#Function that creates a soup ? out of the desired metadata
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

#Create the new soup feature
df['soup'] = df.apply(create_soup, axis=1)

#Display the soup of the first movie
#print(df.iloc[0]['soup'])

#Define a new CountVectorizer object and create vectors for the soup
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df['soup'])

#Since we are using CountVectorizer, cosine_similarity must be used, same as dot product for tf-idf
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

#Reset index of your df and construct reverse mapping again
df = df.reset_index()
indices2 = pd.Series(df.index, index=df['title'])

#Using content_recommender
print(content_recommender('The Lion King', cosine_sim2, df, indices2))

