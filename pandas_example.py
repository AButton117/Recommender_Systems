import pandas as pd
import numpy as np
pd.__version__

#Read the CSV File into df

df = pd.read_csv('data/movies_metadata.csv')

df.head()
#Show type of df
#print(type(df))
#shape of df row x column
#print(df.shape)

#Output the columns of df
#print(df.columns)

#Select the second movie in df
second = df.iloc[1]
#print(second)

#Change the index to the title
df = df.set_index('title')

#Access the movie with the type 'Jumanji'

jum = df.loc['Jumanji']
#print(jum)

#Reset indexing

df = df.reset_index()

#Create a smaller dataframe with a subset of all features

small_df = df[['title', 'release_date','budget','revenue','runtime','genres']]

#print(small_df.head(15))

#Get information of the data types of each feature
#print(small_df.info())

#====================== numpy ==============



#Function to convert to float manually
def to_float(x):
        try:
            x = float(x)
        except:
            x = np.nan
        return x

#Create a smaller dataframe with a subset of all features
small_df = df[['title', 'release_date','budget','revenue','runtime','genres']]

#Apply the to_float function to all values in the budget column
small_df['budget'] = small_df['budget'].apply(to_float)

#try the data types for all features
small_df.info()

#Convert release_date into pandas datetime format
small_df['release_date'] = pd.to_datetime(small_df['release_date'], errors='coerce')

#Extract year from the datetime
small_df['year'] = small_df['release_date'].apply(lambda x:
                                                  str(x).split('-')[0] if x != np.nan else np.nan)

#Display the DataFrame with the new 'year' feature
#print(small_df.head())

#Sort Dataframe based on release year
small_df = small_df.sort_values('year')
#print(small_df.head())

#Sort by revenue
small_df = small_df.sort_values('revenue', ascending=False)
#print(small_df.head())

#Only select movies that have made at least a bil
new = small_df[small_df['revenue'] > 1e9]
#print(new)

#Select movies that have earned over a bil but had a budget of 150 mil
new2 = small_df[(small_df['revenue'] > 1e9) & (small_df['budget'] < 1.5e8)]
print(new2)
