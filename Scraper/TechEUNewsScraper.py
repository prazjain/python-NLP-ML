import sys
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup


class TechEUNewsScraper:

    def __init__(self):
        self._links = []
        self._posts = []

    @property
    def corpus(self):
        return self._posts

    @staticmethod
    def _get_bsoup(url):
        req = Request(url, headers={'User-agent': 'Mozilla/5.0'})
        page = urlopen(req).read().decode('utf8', 'ignore')
        soup = BeautifulSoup(page, 'lxml')
        return soup

    def get_content(self, content_url):
        soup = self._get_bsoup(content_url)
        article = soup.find('div', class_='entry-content').select('p')
        article = ' '.join(map(lambda p: p.text, article)).replace('\\', '').replace('\"', '').replace('\'', '')
        return article

    def build_corpus(self):
        pages_to_scrape = 20
        self._links = []
        self._posts = []
        for page_index in range(1, pages_to_scrape + 1):
            url = self.get_page_link(page_index)
            # print(url)
            soup = self._get_bsoup(url)
            page_links = soup.find('div', {'role': 'main'}).select('header h3 a')
            for pageLink in page_links:
                try:
                    href = pageLink['href']
                    title = pageLink.string.lower()
                    str = 'tech stories this week'
                    if str in title:
                        continue
                    # print (url)
                    self._posts.append(self.get_content(href))
                    self._links.append(href)
                except:
                    print('Exception occurred : ')
                    raise

    @staticmethod
    def get_page_link(num):
        baseUrl = 'https://tech.eu/news/'
        if num == 0 or num == 1:
            return baseUrl
        else:
            return f'{baseUrl}page/{num}/'
