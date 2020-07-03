import re
import string
import pickle as pk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

stop_words = ['and', 'the', 'to', 'of', 'infps', 'infjs', 'intps', 'intjs', 'entps', 'enfps', 'istps', 'isfps', 'entjs', 'istjs', 'enfjs', 'isfjs', 'estps',
              'esfps', 'esfjs', 'estjs', 'infp', 'infj', 'intp', 'intj', 'entp', 'enfp', 'istp', 'isfp', 'entj', 'istj', 'enfj', 'isfj', 'estp', 'esfp', 'esfj', 'estj']

# Define cleaning strategy
punctuations = re.sub(r'[!?]', '', string.punctuation)


def clean_posts(text):
    # clean urls
    res = re.sub(r'http[^\s]*', '', text)
    # clean numbers and replace with 'number'
    res = re.sub(r'[0-9]+', 'number', res)
    # turn to lowercase
    res = res.lower()
    # clean username and replace with 'user'
    res = re.sub(r'@[0-9a-z]+', 'user', res)
    # clean puctuations except '!', '?'
    res = re.sub('[{:s}]*'.format(punctuations), '', res)
    # turn '!'s to 'EMP'(emphases)
    res = re.sub('[!]', ' EMP', res)
    # turn '?'s to 'QST'(questions)
    res = re.sub('[?]', ' QST', res)
    # combine all blanks into one
    res = re.sub(r'[\s]+', ' ', res)
    # remove redundant space
    return res.strip()


def postVectorizer(text):
    with open('pickles/Vectorizer.pk', 'rb') as pkl:
        Vectorizer = pk.load(pkl)
    with open('pickles/Transformer.pk', 'rb') as pkl:
        Transformer = pk.load(pkl)
    return Transformer.transform(Vectorizer.transform([clean_posts(text)])).toarray()
