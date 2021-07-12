# personality-nlp

Personality tests are a widely popular method of self discovery and understanding one another. For this project, I decided to first explore the connections among different personality typing systems through k-means clustering. The personality systems I used were the Enneagram, which classifies an indiviudal into one of nine types labeled with numbers 1-9, Myers-Briggs, which assigns an individual to a type consisting of four letters corresponding to four different traits——either introverted (I) or extroverted (E), sensing (S) or intuitive (N), thinking (T) or feeling (F), and judging (J) or perceiving (P)——and the Big Five, or OCEAN, which consists of five traits——openness, conscientiousness, extraversion, agreeableness, and neuroticism——for which an individual is scored to determine what percentage of each trait they may possess. I then went on to focus on the Myers-Briggs personality types using a data set from Kaggle and proceeded to train and fine-tune a BERT model to classify an individual's personality type based on the language of their online forum posts.

Packages to Install Before Running the Code:

- numpy
- pandas
- matplotlib
- sklearn
- nltk
- spacy
- bs4
- wordcloud
- csv
- re 
- os
- string
- collections
- codecs
- mpld3

# Exploratory Data Analysis

This EDA was performed on the personality type desciptions of the Enneagram, Myers-Briggs, and Big Five types/traits.

1. Use Beautiful Soup to access personality type descirptions from enneagraminstitute.com (Enneagram), 16personalities.com (MBTI), and verywell.com (Big Five). Wasn't necessary to use Beautiful Soup but was useful for learning purposes and becoming familiar with the package
2. Create a word cloud for each of the personality types (see: https://github.com/rachelngordon/personality-nlp/tree/main/EDA%20%26%20Clustering/WordClouds) after removing stop words, punctuation, and common words such as "personality", "type", etc
3. Perform k-means clustering using the type desciptions and following the steps listed on this blog: http://brandonrose.org/clustering

Clustering results can be seen here: 

It appears that for the most part personality types were still grouped according to their personality typing system; so most enneagram types were grouped together for example. This is likely due to the fact that all enneagram type descirptions came from the same source, while all MBTI descriptions came from a different source, and so on. Additionally, it is interesting how this suggests that the language used to describe each enneagram type is more similar to one another than, say, the language used to describe two analytical types like an Enneagram 1 or 5 and an INTJ or INTP from Myers-Briggs.

In the documentary "Persona: The Dark Truth Behind Personality Tests" Merve Emre, referring to MBTI, states, "To me that's what's so extraordinary about [personality] type. It's not the question of whether it's scientifically valid or reliable or not but rather how it is that these two women managed to create a language that would latch onto you and just never ever let you go. A language you would never be able to shake loose." Perhaps the personality typing system one identifies with most says more about a person than their individual personality type?


# Model Building

Data: https://www.kaggle.com/datasnaek/mbti-type
