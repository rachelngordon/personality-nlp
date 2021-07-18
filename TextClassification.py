
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


## FEATURE EXTRACTION


vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train['posts'])
y_train0 = train['type']
X_val = vectorizer.transform(val['posts'])
y_val = val['type']
X_test = vectorizer.transform(test['posts'])
y_test0 = test['type']

#print(vectorizer.get_feature_names())


'''
## MODELING
## Multiclass One vs Rest

# Gradient Boosting

gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, 
                                max_depth=1, random_state=0).fit(X_train, y_train)

print(gb_clf.score(X_test, y_test))

# initial accuracy: 0.43164362519201227

plot_confusion_matrix(gb_clf, X_test, y_test)  
plt.savefig('gradientboosting_cm.png', dpi=200)
'''


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



## train a classifier to determine if someone is introverted or extroverted

# assign target values (0: introvert, 1: extrovert)
y_train, y_test = define_values('I')


logistic_clf1 = LogisticRegression(random_state=0).fit(X_train, y_train)

# check accuracy and plot confusion matrix
print(logistic_clf1.score(X_test, y_test))

plot_confusion_matrix(logistic_clf1, X_test, y_test)  
plt.savefig('IE_confmat.png', dpi=200)



## train a classifier to determine if someone is sensing or intuitive

# assign target values (0: sensing, 1: intuitive)
y_train, y_test = define_values('S')


logistic_clf2 = LogisticRegression(random_state=0).fit(X_train, y_train)

# check accuracy and plot confusion matrix
print(logistic_clf2.score(X_test, y_test))

plot_confusion_matrix(logistic_clf2, X_test, y_test)  
plt.savefig('SN_confmat.png', dpi=200)



## train a classifier to determine if someone is thinking or feeling

# assign target values (0: thinking, 1: feeling)
y_train, y_test = define_values('T')


logistic_clf3 = LogisticRegression(random_state=0).fit(X_train, y_train)

# check accuracy and plot confusion matrix
print(logistic_clf3.score(X_test, y_test))

plot_confusion_matrix(logistic_clf3, X_test, y_test)  
plt.savefig('TF_confmat.png', dpi=200)



## train a classifier to determine if someone is judging or perceiving

# assign target values (0: judging, 1: perceiving)
y_train, y_test = define_values('J')


logistic_clf4 = LogisticRegression(random_state=0).fit(X_train, y_train)

# check accuracy and plot confusion matrix
print(logistic_clf4.score(X_test, y_test))

plot_confusion_matrix(logistic_clf4, X_test, y_test)  
plt.savefig('JP_confmat.png', dpi=200)