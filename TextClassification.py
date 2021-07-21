
import pandas as pd
import numpy as np
import spacy
from spacy.lang.en import English
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt  
from sklearn.metrics import plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.stem.snowball import SnowballStemmer

'''
## PREPROCESSING


# read in the data set
df = pd.read_csv('mbti_1.csv')
print(df.head())
print(df.shape)


## REMOVE STOPWORDS AND PUNCTUATION

def clean_text(text):

    nlp = English()
    my_doc = nlp(text.lower())

    # create list of word tokens
    token_list = []
    for token in my_doc:
        token_list.append(token.text)

    # create list of word tokens after removing stopwords, punctuation, and unnecessary words
    filtered_sentence = [] 
     
    # remove names of each mbti type, stop words, and punctuation
    types_list = ['intj','intp','entj','entp','infj','infp','enfj','enfp',
            'istj','isfj','estj','esfj','istp','isfp','estp','esfp']

    
    for word in token_list:
        lexeme = nlp.vocab[word]

        if lexeme.is_stop == False and lexeme.is_punct == False and word not in types_list:
            filtered_sentence.append(word) 


    # join the list of tokens into a string
    cleaned_text = ' '.join(filtered_sentence)

    # remove URLS
    regex = "http[s]?:\/\/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    cleaned_text = re.sub(regex, '', cleaned_text)
    
    return cleaned_text



for i in range(len(df['posts'])):
    df['posts'][i] = clean_text(df['posts'][i])

print(df['posts'])


## TRAIN-TEST SPLITTING

# shuffle the data and create taining, validation, and test sets
df_shuffled = df.iloc[np.random.permutation(len(df))]
print("shuffled:")
print(df_shuffled.head())
print(len(df_shuffled))


## indexing and splitting of shuffled data

num_train = int(len(df_shuffled)*.7)
val_ind = num_train + int(len(df_shuffled)*.15)
test_ind = val_ind + int(len(df_shuffled)*.15) + 1

# check to make sure this is equal to the total number of observations in the data set
print(test_ind)

train = df_shuffled[0:num_train]
val = df_shuffled[num_train:val_ind]
test = df_shuffled[val_ind:test_ind]

# check again to make sure this is equal to the total number of observations in the data set
print(len(train) + len(val) + len(test))


# save the three separate data sets to csv files
train.to_csv('train.csv', index = False)
val.to_csv('val.csv', index = False)
test.to_csv('test.csv', index = False)

'''


train = pd.read_csv('train.csv')
#print(train.head())
val = pd.read_csv('val.csv')
#print(val.head())
test = pd.read_csv('test.csv')
#print(test.head())


# define a function to tokenize and stem the data

stemmer = SnowballStemmer("english")

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



## FEATURE EXTRACTION


vectorizer = TfidfVectorizer(tokenizer = tokenize_and_stem, max_df = 4500, min_df = 6)
X_train = vectorizer.fit_transform(train['posts'])
y_train0 = train['type']
X_val = vectorizer.transform(val['posts'])
y_val0 = val['type']
X_test = vectorizer.transform(test['posts'])
y_test0 = test['type']

#print(vectorizer.get_feature_names())



## MODELING
# Gradient Boosting

gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, 
                                max_depth=1, random_state=0).fit(X_train, y_train0)


# use validation set to determine what the best min and max document frequencies are
#print(gb_clf.score(X_val, y_val0))
# 0.44273635664873173

# determine accuracy on the test set
print(gb_clf.score(X_test, y_test0))
plot_confusion_matrix(gb_clf, X_test, y_test0)  
plt.savefig('gradientboosting_cm.png', dpi=200)



# defines new target y values to create a binary classification problem 
# based on one given personality trait

def define_values(trait):

    # list of all mbti types
    mbti_types = ['INTJ','INTP','ENTJ','ENTP','INFJ','INFP','ENFJ','ENFP',
                'ISTJ','ISFJ','ESTJ','ESFJ','ISTP','ISFP','ESTP','ESFP']

    # dictionary of regular expressions for finding all types with a given personality trait
    regex_dict={"I":"I[A-Z][A-Z][A-Z]", "E":"E[A-Z][A-Z][A-Z]", "S":"[A-Z]S[A-Z][A-Z]", 
                "N":"[A-Z]N[A-Z][A-Z]", "T":"[A-Z][A-Z]T[A-Z]", "F":"[A-Z][A-Z]F[A-Z]", 
                "J":"[A-Z][A-Z][A-Z]J", "P":"[A-Z][A-Z][A-Z]P"}


    y_train = []
    y_test = []

    for i in range(len(y_train0)):
        list_types = re.findall(regex_dict[trait], str(mbti_types))
        if y_train0[i] in list_types:
            y_train.append(0)
        else:
            y_train.append(1)

    for i in range(len(y_test0)):
        list_types = re.findall(regex_dict[trait], str(mbti_types))
        if y_test0[i] in list_types:
            y_test.append(0)
        else:
            y_test.append(1)

    
    return y_train, y_test




## train a binary classifier to determine one of the four MBTI traits
def train_model(trait1, trait2):

    # assign target values to each personality trait
    y_train, y_test = define_values(trait1)


    logistic_clf = LogisticRegression(random_state=0).fit(X_train, y_train)

    y_pred = logistic_clf.predict(X_test).tolist()

    # check accuracy and plot confusion matrix
    print(logistic_clf.score(X_test, y_test))

    plot_confusion_matrix(logistic_clf, X_test, y_test) 
    doc_name = trait1 + trait2 + '_confmat.png' 
    plt.savefig(doc_name, dpi=200)

    return y_pred


y_pred1 = train_model('I', 'E')
y_pred2 = train_model('S', 'N')
y_pred3 = train_model('T', 'F')
y_pred4 = train_model('J', 'P')


# create a list of the full predicted personality types
y_pred =[]
for i in range(len(y_pred1)):
    type = ''
    if y_pred1[i] == 0:
        type += 'I'
    else:
        type += 'E'
    if y_pred2[i] == 0:
        type += 'S'
    else:
        type += 'N'
    if y_pred3[i] == 0:
        type += 'T'
    else:
        type += 'F'
    if y_pred4[i] == 0:
        type += 'J'
    else:
        type += 'P'

    y_pred.append(type)



# calculate accuracy for the combination of all four binary classifiers

num_correct = 0
for i in range(len(y_pred)):
    if y_pred[i] == y_test0[i]:
        num_correct +=1

accuracy = num_correct/len(y_test0)
print(accuracy)

