#preliminaries for all questions
candidateno=11111113 #this MUST be updated to your candidate number so that you get a unique data sample

#preliminary imports
import sys
sys.path.append(r'\\ad.susx.ac.uk\ITS\TeachingResources\Departments\Informatics\LanguageEngineering\resources')
sys.path.append(r'/Users/juliewe/Documents/teaching/NLE2018/resources')

import re
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from itertools import zip_longest
from nltk.tokenize import word_tokenize
from sussex_nltk.corpus_readers import AmazonReviewCorpusReader
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import math
import spacy
nlp=spacy.load('en')
from nltk.corpus import gutenberg
#question 1


#Do NOT change the code in this cell.

#preparing corpus
def display_sent(asent):
    headings=["token","lower","lemma","pos","NER"]
    info=[]
    for t in asent:
        info.append([t.text,t.lower_,t.lemma_,t.pos_,t.ent_type_])
    return(pd.DataFrame(info,columns=headings))

def tag_sent(asent):
    tagged=[]
    for t in asent:
        tagged.append((t.lower_,t.pos_))
    return tagged

def clean_text(astring):
    #replace newlines with space
    newstring=re.sub("\n"," ",astring)
    #remove title and chapter headings
    newstring=re.sub("\[[^\]]*\]"," ",newstring)
    newstring=re.sub("VOLUME \S+"," ",newstring)
    newstring=re.sub("CHAPTER \S+"," ",newstring)
    newstring=re.sub("\s\s+"," ",newstring)
    #return re.sub("([^\.|^ ])  +",r"\1 .  ",newstring).lstrip().rstrip()
    return newstring.lstrip().rstrip()

def display_tags(sentslist):
    taglist={}
    for s in sentslist:
        for t in s:
            form=t.lower_
            pos=t.pos_
            taglist[pos]=taglist.get(pos,0)+1
    print(len(taglist.keys()))
    print(taglist)

def get_train_test(sentslist,seed=candidateno):
    random.seed(seed)
    random.shuffle(sentslist)
    testsize=10
    train=[tag_sent(s) for s in sentslist[testsize:]]
    test=[tag_sent(s) for s in sentslist[:testsize]]
    return train,test

alice=clean_text(gutenberg.raw('carroll-alice.txt'))
nlp_alice=list(nlp(alice).sents)

#do not change the code in this cell
allsents=list(nlp_alice)
train,test=get_train_test(allsents)

#do not change the code in this cell
class unigram_tagger():

    def __init__(self,trainingdata=[]):
        self.tagcounts={}
        self.wordcounts={}
        self.tagperwordcounts={}
        self.train(trainingdata=trainingdata)

    def train(self,trainingdata):

        for sentence in trainingdata:
            for token,tag in sentence:
                self.tagcounts[tag]=self.tagcounts.get(tag,0)+1
                self.wordcounts[token]=self.wordcounts.get(token,0)+1
                current=self.tagperwordcounts.get(token,{})
                current[tag]=current.get(tag,0)+1
                self.tagperwordcounts[token]=current


#do not change the code in this cell
class hmm_tagger():

    def __init__(self,trainingdata=[]):

        self.emissions={}
        self.transitions={}
        self.train(trainingdata=trainingdata)

    def train(self,trainingdata):

        for sentence in trainingdata:
            previous="START"
            for token,tag in sentence:

                current=self.emissions.get(tag,{})
                current[token]=current.get(token,0)+1
                self.emissions[tag]=current
                bigram=self.transitions.get(previous,{})
                bigram[tag]=bigram.get(tag,0)+1
                self.transitions[previous]=bigram
                previous=tag

        self.emissions={tag:{token:freq/sum(countdict.values()) for (token,freq) in countdict.items()}for (tag,countdict) in self.emissions.items()}
        self.transitions={tag:{token:freq/sum(countdict.values()) for (token,freq) in countdict.items()}for (tag,countdict) in self.transitions.items()}

#question 2

def normalise(tokenlist):
    tokenlist=[token.lower() for token in tokenlist]
    tokenlist=["NUM" if token.isdigit() else token for token in tokenlist]
    tokenlist=["Nth" if (token.endswith(("nd","st","th")) and token[:-2].isdigit()) else token for token in tokenlist]
    tokenlist=["NUM" if re.search("^[+-]?[0-9]+\.[0-9]",token) else token for token in tokenlist]
    return tokenlist

rcr = ReutersCorpusReader().finance()
rcr.enumerate_sents()

random.seed(candidateno)
samplesize=2000
iterations =100
sentences=[]
for i in range(0,iterations):
    sentences+=[normalise(sent) for sent in rcr.sample_sents(samplesize=samplesize)]
    print("Completed {}%".format(i))
print("Completed 100%")

def generate_features(sentences,window=1):
    mydict={}
    for sentence in sentences:
        for i,token in enumerate(sentence):
            current=mydict.get(token,{})
            features=sentence[max(0,i-window):i]+sentence[i+1:i+window+1]
            for feature in features:
                current[feature]=current.get(feature,0)+1
            mydict[token]=current
    return mydict

#question 3

#Do NOT change the code in this cell.

#preparing corpus

def clean_text(astring):
    #replace newlines with space
    newstring=re.sub("\n"," ",astring)
    #remove title and chapter headings
    newstring=re.sub("\[[^\]]*\]"," ",newstring)
    newstring=re.sub("VOLUME \S+"," ",newstring)
    newstring=re.sub("CHAPTER \S+"," ",newstring)
    newstring=re.sub("\s\s+"," ",newstring)
    #return re.sub("([^\.|^ ])  +",r"\1 .  ",newstring).lstrip().rstrip()
    return newstring.lstrip().rstrip()


def get_sample(sentslist,seed=candidateno):
    random.seed(seed)
    random.shuffle(sentslist)
    testsize=int(len(sentslist)/2)
    return sentslist[testsize:]

persuasion=clean_text(gutenberg.raw('austen-persuasion.txt'))
nlp_persuasion=list(nlp(persuasion).sents)

mysample=get_sample(nlp_persuasion)

type(mysample[0])
#question 4
#no code in Question 4


