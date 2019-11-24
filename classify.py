#!/usr/bin/env python3
import nltk

from Scraper.TechEUNewsScraper import TechEUNewsScraper
from Scraper.BBCNewsScraper import BBCNewsScraper
from Classifiers.ThemeClassifier import ThemeClassifier

nltk.download('punkt')
nltk.download('stopwords')

if __name__ == '__main__':
    ns = TechEUNewsScraper()
    bbcs = BBCNewsScraper()
    print('Building corpus of tech articles...')
    ns.build_corpus()
    print('Building classifier...')
    tc = ThemeClassifier(3)
    tc.build(ns.corpus)
    print('Article Themes identified : ')
    print('----------------------------------')
    print(tc.themes)
    print('----------------------------------')
    while True:
        print('Input a tech article link that you will like to classify')
        print('Link should be from https://www.bbc.co.uk/news/ site')
        link = input('Enter a tech article link (empty to exit)>')
        if not link:
            print('Thanks for using this Classifier')
            print('Good Day')
            break
        else:
            content = bbcs.get_content(link)
            theme_identified = tc.predict(content)
            print(f'Article belongs to theme : {tc.themes[theme_identified]}')
