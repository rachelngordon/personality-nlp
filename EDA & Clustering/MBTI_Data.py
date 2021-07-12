
import requests
from bs4 import BeautifulSoup
import spacy
from spacy.lang.en import English
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from string import punctuation
import csv


# get type descriptions from 16Personalities.com

def get_description(mbti_type, url_ext):

    url = 'https://www.16personalities.com/' + mbti_type + url_ext
    type_data = requests.get(url)
    soup = BeautifulSoup(type_data.text, 'html.parser')
    paragraphs = soup.find_all('p')

    all_paragraphs = []
    for paragraph in paragraphs:
        all_paragraphs.append(paragraph.text)

    return all_paragraphs


## lists of all mbti types

types_list = ['intj','intp','entj','entp','infj','infp','enfj','enfp',
            'istj','isfj','estj','esfj','istp','isfp','estp','esfp']

type_names = ['architect','logician','commander','debater', 'advocate','mediator','protagonist',
                'campaigner','logistician','defender','executive','consul','virtuoso','adventurer',
                'entrepreneur','entertainer']

# create a list of plural type names
plural_names = []
for name in type_names:
    plural = name + 's'
    plural_names.append(plural)



# iterate through mbti types and save descriptions from multiple urls in a dictionary

type_descriptions = {}

for mbti_type in types_list:

    type_descriptions[mbti_type] = get_description(mbti_type, url_ext = '-personality')

    url_extensions = ['-strengths-and-weaknesses','-conclusion']
    for str in url_extensions:
        type_descriptions[mbti_type] = type_descriptions[mbti_type] + get_description(mbti_type, str)

    type_descriptions[mbti_type] = ' '.join(type_descriptions[mbti_type]).lower()



## REMOVE STOPWORDS AND PUNCTUATION

def clean_text(text, rm_types):

    nlp = English()
    my_doc = nlp(text)

    # create list of word tokens
    token_list = []
    for token in my_doc:
        token_list.append(token.text)

    # create list of word tokens after removing stopwords, punctuation, and unnecessary words
    filtered_sentence = [] 
     
    remove_tokens = ['personality','personalities','type','types','self','people','need','|','\xa0']
    # remove names of each mbti type when specified
    if rm_types == True:
        remove_tokens = remove_tokens + types_list + type_names + plural_names
    
    for word in token_list:
        lexeme = nlp.vocab[word]

        if lexeme.is_stop == False and lexeme.is_punct == False and word not in remove_tokens:
            filtered_sentence.append(word) 


    # join the list of tokens into a string
    cleaned_text = ' '.join(filtered_sentence)

    # remove numbers
    cleaned_text = ''.join([i for i in cleaned_text if not i.isdigit()])

    # some of the words that were replaced left an s so remove those
    cleaned_text = cleaned_text.replace(' s ', '')

    
    return cleaned_text



# clean text for each type description
for mbti_type in type_descriptions.keys():
    type_descriptions[mbti_type] = clean_text(type_descriptions[mbti_type], rm_types = False)



## WORD CLOUDS

for mbti_type in type_descriptions.keys():

    # create and generate a word cloud image
    wordcloud = WordCloud().generate(type_descriptions[mbti_type])

    # display the generated image
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    #plt.show()

    # save the word cloud image
    file_dest = 'WordClouds/' + mbti_type + '.png'
    wordcloud.to_file(file_dest)



## KEYWORD EXTRACTION
# https://betterprogramming.pub/extract-keywords-using-spacy-in-python-4a8415478fbf

nlp = spacy.load("en_core_web_sm")


def get_hotwords(text):
    result = []
    pos_tag = ['ADJ']
    doc = nlp(text.lower())
    for token in doc:
    
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
            continue
     
        if(token.pos_ in pos_tag):
            result.append(token.text)
                
    return result 


for mbti_type in type_descriptions.keys():
    print(mbti_type)
    hashtags = [('#' + x[0]) for x in Counter(set(get_hotwords(type_descriptions[mbti_type]))).most_common(25)]
    print(' '.join(hashtags))



# clean text for clustering
for mbti_type in type_descriptions.keys():
    type_descriptions[mbti_type] = clean_text(type_descriptions[mbti_type], rm_types = True)



# write the clean data to a csv file

with open('mbti_descriptions.csv', mode='w') as new_file:

    writer = csv.writer(new_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Type', 'Text'])

    for mbti_type in type_descriptions.keys():
        writer.writerow([mbti_type, type_descriptions[mbti_type]])