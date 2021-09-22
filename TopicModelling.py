# Let's read in our document-term matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from nltk import word_tokenize, pos_tag
import scipy.sparse
from gensim import matutils, models
import pandas as pd
import pickle

data = pd.read_pickle('dtm_stop.pkl')
data_clean = pd.read_pickle('data_clean.pkl')
cv = pickle.load(open("cv_stop.pkl", "rb"))


tdm = data.transpose()
sparse_counts = scipy.sparse.csr_matrix(tdm)
corpus = matutils.Sparse2Corpus(sparse_counts)

id2word = dict((v, k) for k, v in cv.vocabulary_.items())

lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=2, passes=50)
lda.print_topics()

lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=3, passes=50)
lda.print_topics()

lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=4, passes=50)
lda.print_topics()


def nouns(text):
    '''Given a string of text, tokenize the text and pull out only the nouns.'''
    def is_noun(pos): return pos[:2] == 'NN'
    tokenized = word_tokenize(text)
    all_nouns = [word for (word, pos) in pos_tag(tokenized) if is_noun(pos)]
    return ' '.join(all_nouns)


data_nouns = pd.DataFrame(data_clean.speeches.apply(nouns))

# Re-add the additional stop words since we are recreating the document-term matrix
add_stop_words = ['like', 'im', 'know', 'just', 'dont', 'thats', 'right', 'people',
                  'youre', 'got', 'gonna', 'time', 'think', 'yeah', 'said']
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

# Recreate a document-term matrix with only nouns
cvn = CountVectorizer(stop_words=stop_words)
data_cvn = cvn.fit_transform(data_nouns.speeches)
data_dtmn = pd.DataFrame(data_cvn.toarray(), columns=cvn.get_feature_names())
data_dtmn.index = data_nouns.index

# Create the gensim corpus
corpusn = matutils.Sparse2Corpus(
    scipy.sparse.csr_matrix(data_dtmn.transpose()))

# Create the vocabulary dictionary
id2wordn = dict((v, k) for k, v in cvn.vocabulary_.items())
# Let's start with 2 topics
ldan = models.LdaModel(corpus=corpusn, num_topics=2,id2word=id2wordn, passes=50)

ldan.print_topics()
# Let's try topics = 3
ldan = models.LdaModel(corpus=corpusn, num_topics=3,id2word=id2wordn, passes=50)

ldan.print_topics()
# Let's try 4 topics
ldan = models.LdaModel(corpus=corpusn, num_topics=4,id2word=id2wordn, passes=50)
ldan.print_topics()

# Let's create a function to pull out nouns from a string of text
def nouns_adj(text):
    '''Given a string of text, tokenize the text and pull out only the nouns and adjectives.'''
    def is_noun_adj(pos): return pos[:2] == 'NN' or pos[:2] == 'JJ'
    tokenized = word_tokenize(text)
    nouns_adj = [word for (word, pos) in pos_tag(
        tokenized) if is_noun_adj(pos)]
    return ' '.join(nouns_adj)


data_nouns_adj = pd.DataFrame(data_clean.speeches.apply(nouns_adj))


cvna = CountVectorizer(stop_words=stop_words, max_df=.8)
data_cvna = cvna.fit_transform(data_nouns_adj.speeches)
data_dtmna = pd.DataFrame(
    data_cvna.toarray(), columns=cvna.get_feature_names())
data_dtmna.index = data_nouns_adj.index


# Create the gensim corpus
corpusna = matutils.Sparse2Corpus(
    scipy.sparse.csr_matrix(data_dtmna.transpose()))

# Create the vocabulary dictionary
id2wordna = dict((v, k) for k, v in cvna.vocabulary_.items())


# Let's start with 2 topics
ldana = models.LdaModel(corpus=corpusna, num_topics=2,
                        id2word=id2wordna, passes=50)
ldana.print_topics()


# Let's try 3 topics
ldana = models.LdaModel(corpus=corpusna, num_topics=3,
                        id2word=id2wordna, passes=50)
ldana.print_topics()


# Let's try 4 topics
ldana = models.LdaModel(corpus=corpusna, num_topics=4,
                        id2word=id2wordna, passes=50)
ldana.print_topics()


# Our final LDA model (for now)
ldana = models.LdaModel(corpus=corpusna, num_topics=4,
                        id2word=id2wordna, passes=80)
print(ldana.print_topics())


corpus_transformed = ldana[corpusna]
print(list(zip([a for [(a, b)] in corpus_transformed], data_dtmna.index)))
