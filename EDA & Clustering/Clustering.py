
## CLUSTERING
# source: http://brandonrose.org/clustering

import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
import spacy
from spacy.lang.en import English
from nltk.stem.snowball import SnowballStemmer


# read in the data
en_types = pd.read_csv('enneagram_descriptions.csv')
mbti_types = pd.read_csv('mbti_descriptions.csv')
ocean_traits = pd.read_csv('ocean_descriptions.csv')

all_types = pd.concat([en_types, mbti_types, ocean_traits])


def remove_words(text):

    nlp = English()
    my_doc = nlp(text)

    # create list of word tokens
    token_list = []
    for token in my_doc:
        token_list.append(token.text)

    # create list of word tokens after removing stopwords, punctuation, and unnecessary words
    filtered_sentence = [] 
    
    # remove common words that do not indicate much about personality type
    remove_words = ['profile','test','result','people','level','view','need','big','tend','step','takes','member','free','entering','things','problems','high','new','suggests','point','results','traits','completely','taking']
    
    for word in token_list:
        if word not in remove_words:
            filtered_sentence.append(word) 


    # join the list of tokens into a string
    cleaned_text = ' '.join(filtered_sentence)

    return cleaned_text


# create a dictionary from the dataframe of all personality types
type_descriptions = {}
i = 0
for type in all_types['Type']:
    type_descriptions[type] = all_types.iloc[i]['Text']
    i += 1

# remove common words from type descriptions
for type in type_descriptions.keys():
    type_descriptions[type] = remove_words(type_descriptions[type])

# convert the dictionary back to a dataframe
type_descriptions = pd.DataFrame(list(type_descriptions.items()),columns = ['Type','Text'])


## STEMMING & TOKENIZING


stemmer = SnowballStemmer("english")


# define a tokenizer and stemmer which returns the set of stems in the text that it is passed

def tokenize_and_stem(text):

    # first tokenize by sentence, then by word
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []

    # filter out any tokens not containing letters
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)

    stems = [stemmer.stem(t) for t in filtered_tokens]

    return stems


def tokenize_only(text):

    # first tokenize by sentence, then by word
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []

    # filter out any tokens not containing letters
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)

    return filtered_tokens


# create total vocabularies of stems and tokens
totalvocab_stemmed = []
totalvocab_tokenized = []

for i in type_descriptions['Text']:
    allwords_stemmed = tokenize_and_stem(i) 
    totalvocab_stemmed.extend(allwords_stemmed) 
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)



# create a dataframe to determine the words in the vocabulary from a stem
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')



from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000, min_df=0.2, use_idf=True, 
                                tokenizer=tokenize_and_stem, ngram_range=(1,3))

# fit the vectorizer to the type descriptions
tfidf_matrix = tfidf_vectorizer.fit_transform(type_descriptions['Text']) 
print(tfidf_matrix.shape)
terms = tfidf_vectorizer.get_feature_names()


from sklearn.metrics.pairwise import cosine_similarity

dist = 1 - cosine_similarity(tfidf_matrix)


## K MEANS

from sklearn.cluster import KMeans

# set the number of clusters
num_clusters = 6

km = KMeans(n_clusters = num_clusters)
km.fit(tfidf_matrix)

clusters = km.labels_.tolist()
print(clusters)


types = {'type': type_descriptions['Type'], 'description': type_descriptions['Text'], 'cluster': clusters}
frame = pd.DataFrame(types, columns = ['type', 'cluster'])

# number of personality types per cluster
print(frame['cluster'].value_counts()) 

# groupby cluster for aggregation purposes
grouped = frame['type'].groupby(frame['cluster'])



# fancy indexing to determine top words for each cluster

print("Top terms per cluster:")
print()
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')
    
    for ind in order_centroids[i, :6]:
        print(' %s' % vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'), end=',')
    print() #add whitespace
    print() #add whitespace
    


# multidimensional scaling

import os  # for os.path.basename

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.manifold import MDS

MDS()

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]
print()
print()


#set up colors per clusters using a dict
#cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3'}
cluster_colors = {0: '#87CEEB', 1: '#008000', 2: '#DAA520', 3: '#6495ED', 4: '#9ACD32', 5: '#CD853F'}

#set up cluster names using a dict
cluster_names = {0: 'Cluster 0',
                 1: 'Cluster 1',
                 2: 'Cluster 2',
                 3: 'Cluster 3',
                 4: 'Cluster 4',
                 5: 'Cluster 5',
                 }

#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=type_descriptions['Type'])) 

#group by cluster
groups = df.groupby('label')


# set up plot
fig, ax = plt.subplots(figsize=(17, 9)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
            label=cluster_names[name],color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          
        which='both',      
        bottom='off',      
        top='off',        
        labelbottom='off')
    ax.tick_params(\
        axis= 'y',         
        which='both',    
        left='off',      
        top='off',        
        labelleft='off')
    
ax.legend(numpoints=1)  #show legend with only 1 point

#add label in x,y position with the label as the film title
for i in range(len(df)):
    ax.text(df.iloc[i]['x'], df.iloc[i]['y'], df.iloc[i]['title'], size=8)  

    
# save plot as png file 
plt.savefig('personality_clusters.png', dpi=200)

