from urllib.request import urlopen, Request
from bs4 import BeautifulSoup


class BBCNewsScraper:

    @staticmethod
    def _base_url():
        return 'https://www.bbc.co.uk/news/'

    @staticmethod
    def _get_bsoup(url):
        req = Request(url, headers={'User-agent': 'Mozilla/5.0'})
        page = urlopen(req).read().decode('utf8', 'ignore')
        soup = BeautifulSoup(page, 'lxml')
        return soup

    def get_content(self, content_url):
        if content_url.startswith(self._base_url()):
            soup = self._get_bsoup(content_url)
            article = soup.find('div', class_='story-body__inner').find_all('p')
            article = ' '.join(map(lambda p: p.text, article)).replace('\\', '').replace('\"', '').replace('\'', '')\
                .replace(':', '')
            return article
        else:
            raise Exception(f'{content_url} should be within {self._base_url}')
