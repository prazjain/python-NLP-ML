from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from string import punctuation
from heapq import nlargest
import nltk


class ThemeClassifier:

    def __init__(self, theme_count):
        self._vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
        # creating clusters using k-means++ algorithm, and exit after 100 iteration if there is not convergence
        self._km = KMeans(n_clusters=theme_count, init='k-means++', max_iter=100, n_init=1, verbose=True)

    @property
    def _ignores(self):
        return ['million', 'billion', 'year', 'millions', 'billions', '\'s', '\"', 'become', 'one', 'let']

    @property
    def themes(self):
        return self._themes

    # Lets extract few themes under which all content below can be categorized
    def build(self, content):

        # fit_transform method takes a list of string, and returns a 2D matrix. Each row represents a document/list item
        self._term_document = self._vectorizer.fit_transform(content)
        # lets find clusters for this corpus
        self._km.fit(self._term_document)
        # lets setup classifier for this content and these clusters
        self._classifier = KNeighborsClassifier(n_neighbors=10)
        self._classifier.fit(self._term_document, self._km.labels_)

        # lets check the themes these documents got categorized into
        text = {}
        for i, cluster in enumerate(self._km.labels_):
            oneDoc = content[i]
            if cluster not in text.keys():
                text[cluster] = oneDoc
            else:
                text[cluster] += ' ' + oneDoc
        _stopwords = set(
            stopwords.words('english') + list(punctuation) + self._ignores)

        # find 100 most important keywords from each cluster
        keywords = {}
        counts = {}
        for cluster in range(3):
            word_sent = word_tokenize(text[cluster].lower())
            word_sent = [word for word in word_sent if word not in _stopwords]
            freq = FreqDist(word_sent)
            keywords[cluster] = nlargest(100, freq, key=freq.get)
            counts[cluster] = freq

        # now find top 10 keywords unique to each cluster
        unique_keys = {}
        for cluster in range(3):
            other_clusters = list(set(range(3)) - set([cluster]))
            keys_other_clusters = set(keywords[other_clusters[0]]).union(set(keywords[other_clusters[1]]))
            unique = set(keywords[cluster]) - keys_other_clusters
            unique_keys[cluster] = nlargest(10, unique, key=counts[cluster].get)

        # print(unique_keys)
        self._themes = unique_keys

    # Now lets predict the theme for content passed here
    def predict(self, content):
        docTfIdf = self._vectorizer.transform([content])
        theme = self._classifier.predict(docTfIdf)
        print(f'Theme is {theme[0]}')
        return theme[0]