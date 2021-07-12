
import requests
from bs4 import BeautifulSoup
import spacy
from spacy.lang.en import English
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from string import punctuation
import csv


# get type descriptions from Enneagram Institute
type_descriptions = {}

for value in range(1,10):

    url = 'https://www.enneagraminstitute.com/type-' + str(value)
    type_data = requests.get(url)
    soup = BeautifulSoup(type_data.text, 'html.parser')
    paragraphs = soup.find_all('p')

    all_paragraphs = []
    for paragraph in paragraphs:
        all_paragraphs.append(paragraph.text)

    
    # remove unnecessary paragraphs
    exclude = ['Examples', 'Level', 'Privacy', 'Healthy', 'Average', 'Unhealthy']
    for paragraph in all_paragraphs:

        for word in exclude:
            if paragraph.startswith(word) == True:
                all_paragraphs.remove(paragraph)
 

    type_descriptions[value] = all_paragraphs


# join the paragraphs into one body of text and convert to all lowercase
for en_type in type_descriptions.keys():
    type_descriptions[en_type] = ' '.join(type_descriptions[en_type])
    type_descriptions[en_type] = type_descriptions[en_type].lower()


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
     
    remove_tokens = ['level','levels','personality','personalities','enneagram','type','types','self','healthy','unhealthy','|','\xa0']
    
    # remove names of each enneagram type when specified
    if rm_types == True:
        remove_tokens = remove_tokens + ['ones','twos','threes','fours','fives','sixes','sevens','eights','nines']
    
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
for en_type in type_descriptions.keys():
    type_descriptions[en_type] = clean_text(type_descriptions[en_type], rm_types = False)




## WORD CLOUDS

for en_type in type_descriptions.keys():

    # create and generate a word cloud image
    wordcloud = WordCloud().generate(type_descriptions[en_type])

    # display the generated image
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    #plt.show()

    # save the word cloud image
    file_dest = 'WordClouds/' + str(en_type) + '.png'
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


for en_type in type_descriptions.keys():
    print(en_type)
    hashtags = [('#' + x[0]) for x in Counter(set(get_hotwords(type_descriptions[en_type]))).most_common(25)]
    print(' '.join(hashtags))



# clean text for clustering
for en_type in type_descriptions.keys():
    type_descriptions[en_type] = clean_text(type_descriptions[en_type], rm_types = True)



# write the clean data to a csv file

with open('enneagram_descriptions.csv', mode='w') as new_file:

    writer = csv.writer(new_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Type', 'Text'])

    for en_type in type_descriptions.keys():
        writer.writerow([str(en_type), type_descriptions[en_type]])