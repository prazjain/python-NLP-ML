# Natural Language Processing with Python

Here we will read text using standard techniques of Natural Language Processing and Machine Learning. 

**Tech Used**

* Python
* Beautiful Soup
* NLTK
* SKLEARN

### Tasks in Natural Language Processing

These are broken down into:

* Tokenization : Breaking down text into words and sentences.
* Stop word removal: Filter out common words that do not add much meaning.
* N Grams : Identify groups of words that commonly occur together. eg of Bi-gram is 'New York'.
* Word sense disambiguation: Identify meaning of word based on the context in which it is occuring.
* Parts of Speech: Identify which words are noun, verb, adverb, adjective etc.
* Stemming: Extract root of a word. eg closed, close, closer, closely etc.

### Types of Machine Learning Problems

* Classification : To classify a text/item into a category. 
    * Given an email, find out if it is spam or not spam.
    * Given a credit card transaction, find out if it is fraud or not.

* Clustering : Given a large group of articles, divide them into groups based on some common attributes. And these new groups are not known beforehand, they are derived from the text contained in the articles.

## Auto Summarize Text

Here I will summarize a BBC news article in below steps:
 1) Find important words
 2) Give score to sentences that contain important words
 3) Pull high score sentences into summary

Run the below commands in the directory of downloaded code : 

    chmod +x summarize.py
    pip3 install bs4
    pip3 install nltk
    pip3 install lxml
    ./summarize.py https://www.bbc.co.uk/news/world-europe-50430855

And you will see the summary of above article in 3 sentences (default summary size) like below : 

_The BBCs Jenny Hill in Venice writes: The first flood sirens went off at dawn, an eerie sound rising over the ancient bridges and waterways of the city._

_It hurts to see the city so damaged, its artistic heritage compromised, its commercial activities on its knees, Prime Minister Giuseppe Conte, who visited Venice on Wednesday, wrote in a Facebook post (in Italian)._

_Mr Conte said the government would accelerate the Mose project - construction of a hydraulic barrier system to protect the lagoon from rising sea levels and winter storms._

Read the article for yourself and see if this summary is good enough for summarizing the news in 3 lines.

You can run the program in below 3 ways :

    ./summarize.py
    ./summarize.py https://www.bbc.co.uk/news/world-europe-50430855
    ./summarize.py https://www.bbc.co.uk/news/world-europe-50430855 5

* First one summarizes a default article : https://www.bbc.co.uk/news/world-europe-50430855 in 3 lines (default).
* Second one will summarize the article chosen by you in 3 lines (default)
* Third one will summarize an article of your choice in number of lines of your chosing (here it is 5).

Try other news items like : 

https://www.bbc.co.uk/news/election-2019-50497288

https://www.bbc.co.uk/news/uk-50506909

Above Text Auto Summarizer only reads bbc news content links, so can only summarize that news, but it can be extended to read other content and summarize them.

## Classify Text Using Machine Learning

Classifying text means, identify themes/category associated with a piece of text.

#### Typical Machine Learning Workflow

A typical workflow includes: 
 1) Identifying the problem (Classification/Clustering)
 2) Representing text data using numeric attributes (Feature Extraction)
 3) Training the model/classifier using training data (Training)
 4) Test the model/classifier using test data (Testing)


##### Feature Extraction in Machine Learning

In this steps we are basically converting our text data into numeric form, so we can do operations on it.

**Term Frequency (Bag of Words Model)**

We create a tuple with a universe of all possible words. eg our universe of all possible words is : 

    hello, this, is, the, universe, of, all, possible, words, goodbye

Now given input string: 

    hello universe goodbye

The tuple will looks like this:

    (1,0,0,0,1,0,0,0,0,1)

Bag of Words model keeps a count of term frequency but does not keep track of order of words, so it loses it.

**Term Frequency - Inverse Document Frequency**

This is an improvement on Term Frequency, so apart from keeping a count of term frequency, we also attach weight to it. Words that occur more often in a document but are rare across other documents, will have more weight, compared to word that occur a lot in all the documents.

    Weight = 1 / # documents in corpus the word appears in

So each word's term frequency in tuple is multiplied by weight of the word. This is Term Frequency - Inverse Document Frequency

**Choice of Algorithm**

Choice of algorithm, depends on the type of problem we are trying to solve.
* Classification
    
    * Naive Bayes
    * Support Vector Machines

* Clustering

    * K-Means Clustering
    * Hierarchical Clustering

**K Means Clustering**

We have our documents represented in TF - IDF.

Each document is a tuple of N numbers. N is the total number of distinct words in all the documents. So a document is now a point in N dimensional hypercube space.

When we do clustering, we minimize the distance between points in same cluster, and maximize the distance between points that are in different cluters. This is done with K Means Clustering

**Steps in K Means Clustering Algorithm**

1. Initialize a set of points as 'K' means. (Centroids of the clusters that you want to find). 'K' is the number of clusters that you want to divide the data into. Each of these points is a centroid of the cluster.
2. Every point in N dimensional hypercube is assigned to the cluster belonging to the nearest mean. So al documents are now grouped into K clusters.
3. Take mean of all coordinates that are in a cluster, we can find new mean/centroid of the cluster.
4. Repeat step 2,3 until means don't change anymore. This point is called _Convergence_. Sometimes when there is no convergence, then we can set maximum number of iterations of the algorithms to be performed. 

At the end of this, we will have a set of K groups under which all points (documents) have been assigned.

### K Nearest Neighbours

There are several algorithms to choose from for creating Classifier Model. One of them is _K Nearest Neighbours_

1. From clustering steps we have articles grouped in themes/group.
2. When we get a new article that is also represented in TF-IDF in tuple representation, it it will be another point in N dimensional hypercube.
3. So in _K Nearest Neighbours_ we find K nearest neighbours to this article.
4. We find the theme/group that majority of the K nearest neighbours belong to, and assign that theme to the new article.


### Building our Classifier Model to identify Article theme using Machine Learning

1. We will let computer read lot of tech articles from `https://tech.eu/news/` to build a corpus.
2. Algorithm will identify the K themes in which to group this corpus of data. (K is configurable number)
3. Now we will feed new tech article to computer from another website `https://www.bbc.co.uk/news/technology-50418357`
4. Our classifier model will tell us which of the K themes, does the new article belong to.

Pre-requisites

    chmod +x classify.py
    pip3 install bs4
    pip3 install nltk
    pip3 install lxml
    pip3 install sklearn

Below is output from my run :

        (venv) (base) Prashants-MacBook-Pro:python-NLP-ML Prashant$ ./classify.py
        Building corpus of tech articles...
        Building classifier...
        Initialization complete
        Iteration  0, inertia 237.968
        Iteration  1, inertia 122.334
        Iteration  2, inertia 122.074
        Iteration  3, inertia 122.036
        Iteration  4, inertia 121.956
        Iteration  5, inertia 121.921
        Converged at iteration 5: center shift 0.000000e+00 within tolerance 4.162521e-08
        Article Themes identified : 
        ----------------------------------
        {0: ['🎧', 'podcast', 'markets', 'today', '—', 'tech.eu', 'percent', 'saas', 'entrepreneurs', 'interview'], 1: ['research', 'women', 'medical', 'rare', 'process', 'art', 'treatment', 'analysis', 'see', 'drug'], 2: ['experience', 'service', 'since', 'mobility', 'solutions', 'vision', 'group', 'around', 'venture', 'operations']}
        ----------------------------------
        Input a tech article link that you will like to classify
        Link should be from https://www.bbc.co.uk/news/ site
        Enter a tech article link (empty to exit)>https://www.bbc.co.uk/news/technology-50418357
        Theme is 0
        Article belongs to theme : ['🎧', 'podcast', 'markets', 'today', '—', 'tech.eu', 'percent', 'saas', 'entrepreneurs', 'interview']
        Input a tech article link that you will like to classify
        Link should be from https://www.bbc.co.uk/news/ site
        Enter a tech article link (empty to exit)>https://www.bbc.co.uk/news/technology-50316951
        Theme is 2
        Article belongs to theme : ['experience', 'service', 'since', 'mobility', 'solutions', 'vision', 'group', 'around', 'venture', 'operations']
        Input a tech article link that you will like to classify
        Link should be from https://www.bbc.co.uk/news/ site
        Enter a tech article link (empty to exit)>
        Thanks for using this Classifier
        Good Day
        (venv) (base) Prashants-MacBook-Pro:python-NLP-ML Prashant$ 


eg. After reading around 150 news article from `https://tech.eu/news/` website, we identified below theme categories :

* First theme generated, relates to below concepts/categories:

        ['🎧', 'podcast', 'markets', 'today', '—', 'tech.eu', 'percent', 'saas', 'entrepreneurs', 'interview']

* Second theme generated, relates to these concepts/categories:

        ['research', 'women', 'medical', 'rare', 'process', 'art', 'treatment', 'analysis', 'see', 'drug']

* Third theme generated, relates to these concepts/categories:

        ['experience', 'service', 'since', 'mobility', 'solutions', 'vision', 'group', 'around', 'venture', 'operations']

Now when I feed this next article which is unknown to the machine `https://www.bbc.co.uk/news/technology-50418357`, machine tells me that this article belongs to first theme : 

        ['🎧', 'podcast', 'markets', 'today', '—', 'tech.eu', 'percent', 'saas', 'entrepreneurs', 'interview']

And when I feed it another news article `https://www.bbc.co.uk/news/technology-50316951`, machine tells me it belongs to third theme : 

        ['experience', 'service', 'since', 'mobility', 'solutions', 'vision', 'group', 'around', 'venture', 'operations']

You can go and read the article and judge for yourself which of the 3 themes does this new article belongs to.

And feed different news articles to find the theme they belong to.

(I have also attached Jupyter Notebook files, which I used trying out different things)

Enjoy