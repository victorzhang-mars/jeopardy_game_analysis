import pandas as pd
import spacy
from spacy.lang.en import English
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
import random
from gensim import corpora
import pickle
import gensim
import re
import string

spacy.load('en_core_web_sm')
#nltk.download('wordnet')

parser = English()


def tokenize(text):
    lda_tokens = []
    # process text through english model of spacy
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


def get_lemma(word):
    """
    >>> print(wn.morphy('dogs'))
    dog
    >>> print(wn.morphy('churches'))
    church
    >>> print(wn.morphy('aardwolves'))
    aardwolf
    """
    lemma = wn.morphy(word)

    if lemma is None:
        # return the original one
        return word
    else:

        return lemma


def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)


#nltk.download('stopwords')
# english stop words
en_stop = set(nltk.corpus.stopwords.words('english'))


def prepare_text_for_lda(text):
    # tokenize
    tokens = tokenize(text)
    # filter word length > 4
    tokens = [token for token in tokens if len(token) > 4]
    # stopwords
    tokens = [token for token in tokens if token not in en_stop]
    return tokens

def words(text):
    """
    Given a string, return a list of words normalized as follows.
    Split the string to make words first by using regex compile() function
    and string.punctuation + '0-9\\r\\t\\n]' to replace all those
    char with a space character.
    Split on space to get word list.
    Ignore words < 3 char long.
    Lowercase all words
    Remove English stop words
    """
    # tStart = time.time()
    parr_ENGLISH_STOP_WORDS = [
        "a", "about", "above", "across", "after", "afterwards", "again", "against",
        "all", "almost", "alone", "along", "already", "also", "although", "always",
        "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
        "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
        "around", "as", "at", "back", "be", "became", "because", "become",
        "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
        "below", "beside", "besides", "between", "beyond", "bill", "both",
        "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
        "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
        "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
        "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
        "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
        "find", "fire", "first", "five", "for", "former", "formerly", "forty",
        "found", "four", "from", "front", "full", "further", "get", "give", "go",
        "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
        "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
        "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
        "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
        "latterly", "least", "less", "ltd", "made", "many", "may", "me",
        "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
        "move", "much", "must", "my", "myself", "name", "namely", "neither",
        "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
        "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
        "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
        "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
        "please", "put", "rather", "re", "same", "see", "seem", "seemed",
        "seeming", "seems", "serious", "several", "she", "should", "show", "side",
        "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
        "something", "sometime", "sometimes", "somewhere", "still", "such",
        "system", "take", "ten", "than", "that", "the", "their", "them",
        "themselves", "then", "thence", "there", "thereafter", "thereby",
        "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
        "third", "this", "those", "though", "three", "through", "throughout",
        "thru", "thus", "to", "together", "too", "top", "toward", "towards",
        "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
        "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
        "whence", "whenever", "where", "whereafter", "whereas", "whereby",
        "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
        "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
        "within", "without", "would", "yet", "you", "your", "yours", "yourself",
        "yourselves"]

    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    # delete stuff but leave at least a space to avoid clumping together
    nopunct = regex.sub(" ", text)
    words = nopunct.split(" ")
    # words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    # words = [w for w in words if len(w) > 2]  # ignore a, an, to, at, be, ...
    # words = [w.lower() for w in words]
    words = [w.lower() for w in words if len(
        w) > 2 and w not in parr_ENGLISH_STOP_WORDS]
    # print words
    # tEnd = time.time()
    # print(f"words() cost {tEnd - tStart} secs")
    return ' '.join(words)

def preprocess(df):
    """only add more columns, will not decrease, return df
    """
    # data cleaning, rename column names so that they can be called in oop way
    dct_rename = dict(
        zip(df.columns, [i.strip().replace(' ', '_').lower() for i in df.columns]))
    df = df.rename(columns=dct_rename)

    # make one more feature 'year', so we can break them up by year
    df['year'] = df['air_date'].apply(lambda x: x.split('-')[0])

    # make one more feature 'round_enc' as the numerical representation of column 'round'
    mapping_round = dict(
        zip(df['round'].unique().tolist(), range(len(df['round'].unique().tolist()))))
    df['round_enc'] = df['round'].map(mapping_round)

    # combine the category into question
    df['cat_question'] = df['category'] + ', ' + df['question']

    # more string processing
    df['cat_question'] = df['cat_question'].apply(words)

    return df


def sub_set(df, lst_year, rnd):
    """return df[f1 & f2]
    """
    f1 = df['year'].isin(lst_year)
    f2 = df['round_enc'] == rnd
    return df[f1 & f2]


def prepare_data(df_sub):
    text_data = []
    # print(df_sub.cat_question.to_list())
    # with open('Machine-Learning-with-Python/dataset.csv') as f:
    for line in df_sub.cat_question.to_list():
        tokens = prepare_text_for_lda(line)
        random.seed(42)
        # if random.random() > .95:
        if random.random() > .5:
            # print(tokens)
            text_data.append(tokens)
    # print(text_data)
    return text_data


def build_corpus(text_data):
    # gensim get the dictionary
    dictionary = corpora.Dictionary(text_data)
    # build corpus
    corpus = [dictionary.doc2bow(text) for text in text_data]
    return dictionary, corpus


def train(k, corpus, dictionary):
    # find k topics through LDA
    NUM_TOPICS = k
    ldamodel = gensim.models.ldamodel.LdaModel(
        corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=15, random_state=42)
    # ldamodel.save('model5.gensim')

    topics = ldamodel.print_topics(num_words=10)
    # # for topic in topics:
    # print(topic)
    return topics


def Q1_gensim(df, year_group, rnd, k):
    df_sub = sub_set(df, year_group, rnd)
    text_data = prepare_data(df_sub)
    dictionary, corpus = build_corpus(text_data)
    topics = train(k, corpus, dictionary)
    return topics


if __name__ == '__main__':
    # load the data
    df = pd.read_csv('JEOPARDY_CSV.csv')

    df = preprocess(df)

    year_group1 = [str(i) for i in range(1984, 1994)]
    year_group2 = [str(i) for i in range(1994, 2004)]
    year_group3 = [str(i) for i in range(2004, 2013)]

    # # save the dictionary for future uses
    # pickle.dump(corpus,open('corpus.pkl','wb'))
    # dictionary.save('dictionary.gensim')
