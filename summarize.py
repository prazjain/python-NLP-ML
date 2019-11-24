#!/usr/bin/env python3
import sys
import nltk

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

from string import punctuation
from heapq import nlargest
from collections import defaultdict

from Scraper.BBCNewsScraper import BBCNewsScraper

nltk.download('punkt')
nltk.download('stopwords')


def summ(text, n):
    sents = sent_tokenize(text)
    assert n <= len(sents)
    word_sent = word_tokenize(text.lower())
    _stopwords = set(stopwords.words('english') + list(punctuation))
    word_sent = [word for word in word_sent if word not in _stopwords]
    freq = FreqDist(word_sent)
    ranking = defaultdict(int)
    for sent_idx, sent in enumerate(sents):
        for word in word_tokenize(sent.lower()):
            if word in freq:
                ranking[sent_idx] += freq[word]
    top_idx = nlargest(n, ranking, key=ranking.get)
    return [sents[idx] for idx in sorted(top_idx)]


if __name__ == '__main__':
    count = len(sys.argv)
    url = 'https://www.bbc.co.uk/news/world-europe-50430855'
    summary_size = 3
    if count == 3:
        summary_size = int(sys.argv[2])
    if count >= 2:
        url = sys.argv[1]

    ns = BBCNewsScraper()

    summary = summ(ns.get_content(url), summary_size)

    print('-------------------------------------------')
    print('Summary')
    print('-------------------------------------------')
    for sentence in summary:
        print(sentence)
    print('-------------------------------------------')
    print('Usage :')
    print('./summarize.py <bbc_news_link> <number>')
    print('<bbc_news_link>: Any BBC News link that starts with : https://www.bbc.co.uk/news/')
    print('<number>: Number of lines in which you want to summarize the article. 3 is default. Try 2,4,7 etc..')
    print('-------------------------------------------')
