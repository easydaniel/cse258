import pickle
import gzip
import numpy as np
import urllib
import scipy.optimize
import random
from collections import defaultdict
import string
import nltk
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer



class Draw:
    category = dict()
    categoryNames = ['comics_graphic', 'children', 'romance', 'young_adult', 'poetry', 'fantasy_paranormal', 'history_biography', 'mystery_thriller_crime']
    
    def __init__(self, dataset):
        self.dataset = dataset
        for name in self.categoryNames:
            self.category[name] = [d for d in dataset if d['category'] == name]
        self.description = [d['description'] for d in dataset]
        return

    def drawWordCloud(self, name, wordCount):
        wc = WordCloud(background_color="white", max_words=500, width=1024, height=720,)
        wc.generate_from_frequencies(wordCount)
        plt.figure(figsize=(20,10))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(name, format = 'png')
        plt.show()
        return

    def writeText(self, name, wordlist):
        f = open(name+".txt", 'w')
        for w in wordlist:
            f.writeline(w[0] + " " + str(w[1]))
        f.close()
        return

    def drawBOF(self, name):
        corpus = [d['description'] for d in self.category[name]]
        wordCount = defaultdict(int)
        punctuation = set(string.punctuation)
        stemmer = PorterStemmer()

        for d in corpus:
            r = ''.join([c for c in d.lower() if not c in punctuation])
            for w in r.split():
                w = stemmer.stem(w) # with stemming
                wordCount[w] += 1  
        
        counts = [(wordCount[w], w) for w in wordCount]
        counts.sort()
        counts.reverse()

        self.writeText(name+ '_bof' , [[x[1], x[0]] for x in counts[:30]])
        self.drawWordCloud(name + '_bof' , wordCount)
        return

    def find_ngrams(self, input_list, n):
        return zip(*[input_list[i:] for i in range(n)])
        
    def drawBagsofNgram(self, name, n):
        corpus = [d['description'] for d in self.category[name]]
        wordCount = defaultdict(int)
        punctuation = set(string.punctuation)
        stemmer = PorterStemmer()
        for d in corpus:
            s = ''.join([c for c in d.lower() if not c in punctuation])
            ngram = self.find_ngrams(s.split(), n)
            for i in ngram:
                #w = stemmer.stem(i)
                wordCount['-'.join(i)] += 1
        
        counts = [(wordCount[w], w) for w in wordCount]
        counts.sort()
        counts.reverse()   
        self.writeText(name+"_" +str(n)+"gram", [[x[1], x[0]] for x in counts[:30]])
        self.drawWordCloud(name +"_" +str(n)+"gram", wordCount)    
        return

    def drawTFIDFNgram (self, name, n):
        corpus = [d['description'] for d in self.category[name]]
        vectorizer = TfidfVectorizer(stop_words = 'english', ngram_range = (n, n))
        vectorizer.fit(self.description)
        cattdm = vectorizer.transform(corpus)
        weight = cattdm.sum(axis = 0)
        mapping = vectorizer.vocabulary_
        wordCounts = dict()

        for i in mapping.items():
            #print(i, type(weight[0,i[1]]))
            w = i[0].split()
            wordCounts['-'.join(w)] = weight[0,int(i[1])]

        counts = [(wordCounts[w], w) for w in wordCounts]
        counts.sort()
        counts.reverse()

        self.writeText(name+"_" +str(n)+"gramTFIDF", [[x[1], x[0]] for x in counts[:30]])
        self.drawWordCloud(name+"_" +str(n)+"gramTFIDF", wordCounts)  
        return





if __name__ ==  '__main__':

    dataset = pickle.load(gzip.open('dataset.pickle.gz'))
    draw = Draw(dataset)
    
    draw.drawBOF('comics_graphic')
    draw.drawBagsofNgram('comics_graphic', 2)

    draw.drawTFIDFNgram('children', 2)
