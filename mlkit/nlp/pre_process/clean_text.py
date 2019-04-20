import re
import nltk
import spacy
from nltk.stem import PorterStemmer
import pandas as pd
from nltk.stem.rslp import RSLPStemmer
import re

STEM = "Stemizacao"
LEM = "Lemmatizacao"
    
class TextCleaner:
    
    def __init__(self, lan = 'english', norm = None, load_v = None, clean_regex = "([aA-zZ]+)", stemmer = RSLPStemmer()):
        nltk.download('stopwords', quiet = True)
        self.regex = clean_regex
        self.stops = nltk.corpus.stopwords.words(lan)
        
        if load_v == None:
            if lan == "english":
                self.nlp = spacy.load('en')
            elif lan == "portuguese":
                self.nlp = spacy.load('pt_core_news_sm')
            else:
                raise("idioma invalido")
        else:
            self.nlp = spacy.load(load_v)
        self.ps = stemmer
        
        if (norm != STEM) and (norm != LEM) and (norm != None):
            raise("Normalizacao invalida")
        self.norm = norm
        
    def clean(self, df, stops = None):
        if stops != None:
            self.stops = stops
            
        return df.apply(self.remove_special_char).apply(lambda x: self.remove_stop_words(x, self.stops)).apply(self.normalize)
        
    def remove_special_char(self, text):
        return " ".join(["".join(re.findall(self.regex, word)).lower() for word in text.split(" ")])
    
    def remove_stop_words(self, text, stops):
        if stops != None:
            return " ".join([word if word not in self.stops else "" for word in text.split(" ")])
        else:
            return text
        
    def normalize(self, text):
        text = re.sub(' +', ' ', text)
        text = re.sub('^ | $', '', text)
        if len(text) > 0:
            if self.norm == LEM:
                doc = self.nlp(text)
                return " ".join([token.lemma_ for token in doc])
            elif self.norm == STEM:
                return " ".join([self.ps.stem(word) for word in text.split(" ")])
        return text