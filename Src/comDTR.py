from __future__ import division
from itertools import *
from pylab import *
from matplotlib import pyplot

import pandas as pd
import numpy as np
import pickle
import nltk, collections, re, string, scipy, gc, os
from scipy import sparse
from scipy.sparse import csr_matrix, dia_matrix
from itertools import groupby
from nltk.corpus import stopwords   # stopwords to detect language
from nltk import wordpunct_tokenize # function to split up our words
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import csr_matrix, dia_matrix
from nltk.stem.snowball import SnowballStemmer
#from nltk.stem.porter import PorterStemmer

from gensim.models.word2vec import Word2Vec
from nltk.stem.snowball import SnowballStemmer

class DTR:    
    def removeStops(self, text):
        text = re.sub("^\xa9[^.]*\\. ", "", text)
        text = re.sub("^.*?(A|a)ll rights reserved\\.? ", "", text)
        # # Arreglando textos minuzculas, quitando puntuaciones, asentos, palabras repetidas, duplicados
        cachedStopWords        = stopwords.words("english")
        remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)        
        exclude = set(string.punctuation)   
        result = ' '.join([word.lower() for word in text.translate(remove_punctuation_map).split() 
                        if word not in cachedStopWords and not any([c.isdigit() for c in word])])
        return(result)

    """ Represents a results of DTR"""
    def __init__(self, listDocs, titleOri, abrevFile = "", maxDF = 1.0, minDF = 1, maxDF_ori = 1.0, minDF_ori = 1):
        self.maxDF_ori = maxDF_ori
        self.minDF_ori = minDF_ori
        listDocs = [self.removeStops(ww) for ww in listDocs]
        stemmer = SnowballStemmer("english")
        def tokenize_and_stem(text):
            #first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
            tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
            filtered_tokens = []
            # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
            for token in tokens:
                if re.search('[a-zA-Z]', token):
                    filtered_tokens.append(token)
            stems = [stemmer.stem(t) for t in filtered_tokens]
            return stems
   
        co_Vectorizer = CountVectorizer(max_df = maxDF, min_df = minDF, 
                                        stop_words = 'english',
                                        tokenizer  = tokenize_and_stem)
        self.A = co_Vectorizer.fit_transform(listDocs)
        self.vocabulary = co_Vectorizer.vocabulary_
        print("---- Finish stemming and tokenization\n")
        self.computeDTR()
        self.weightDocs(artOri = titleOri, abrevFile = abrevFile)
                
    def computeDTR(self):
        # # TCOR matrix
        tcooCur         = self.A      
        tcooCur.data[:] = tcooCur.data / tcooCur.data
        
        # # Found DF for terms
        self.nt = tcooCur.sum(axis=0)  
        
        tcooCur         = tcooCur.T * tcooCur
        tcooCur.data[:] = (1 + np.log(tcooCur.data))
        nor_Row         = np.log(len(self.vocabulary))/ np.bincount(tcooCur.nonzero()[0])
        tcooCur         = dia_matrix((nor_Row, np.array([0])), shape=(len(nor_Row), len(nor_Row))) * tcooCur 
        self.TCOR       = tcooCur
        
        del tcooCur, nor_Row
        gc.collect()
        
        # # DOR matrix
        dcor         = self.A.T
        dcor.data[:] = (1 + np.log(dcor.data))
        nor_Row      = np.log(len(self.vocabulary)) / np.bincount(dcor.nonzero()[1])
        dcor         = dcor * dia_matrix((nor_Row, np.array([0])), shape=(len(nor_Row), len(nor_Row)))
        self.DOR     = dcor
    
    def weightDtr(self, texts, tfs, vocabulary, flagTCOR = True):
        result = []
        for row, text in texts:
            mt_row = tfs.getrow(row)
            index    = [(self.vocabulary.get(ww), mt_row.getcol(vocabulary[ww]).data[0]) for ww in text 
                        if not self.vocabulary.get(ww) is None]
            nrow  = len(index)
            alpha = csr_matrix(([jj for ii, jj in index], (np.repeat(0, nrow), range(nrow))), shape=(1, nrow))
            if flagTCOR:
                result.append(alpha * self.TCOR[np.array([ii for ii, jj in index]), :])
            else:
                result.append(alpha * self.DOR[np.array([ii for ii, jj in index]), :])
                
        return(scipy.sparse.vstack(result))
    def plotDF(self):
        # Histogram of numbers of Words
        pyplot.subplot(311)
        counts,bin_edges = np.histogram(np.bincount(np.sort(self.A.nonzero()[0])), 20)
        bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
        err = np.random.rand(bin_centres.size)*100
        pyplot.semilogy(bin_centres, counts, 'ro')
        title("Frequency distribution of document")
        xlabel("Numbers of Words")
        ylabel("Frequency")

        # A log-log plot to Document Frequency
        pyplot.subplot(312)
        freq_DF = collections.Counter(np.array(self.nt)[0])
        counts  = collections.OrderedDict(sorted(freq_DF.items()))
        frequencies = [v for k, v in counts.iteritems()]
        ranks       = [k for k, v in counts.iteritems()]
        loglog(range(len(frequencies)), frequencies, marker=".")
        title("DF Scopus Corpus")
        xlabel("Document Frequency")
        ylabel("Numbers of Words")
        grid(True)
    
    def weightDocs(self, artOri, abrevFile = ""):
        # # Compute WTCOR      
        texts, parseTexts, tfidf_vocabulary, mat_tfidf = makeTFidf(artOri, max_df = self.maxDF_ori, min_df=self.minDF_ori)
        WTCOR = self.weightDtr(parseTexts, mat_tfidf, tfidf_vocabulary.vocabulary_, flagTCOR = True)
        self.TCOR = WTCOR
        print(WTCOR.asformat)

        # # Compute DOR
        WDOR = self.weightDtr(parseTexts, mat_tfidf, tfidf_vocabulary.vocabulary_, flagTCOR = False)
        self.WDOR = WDOR
        print(WDOR.asformat)
        self.titleOri = texts
        self.tfidfOri = tfidf_vocabulary
        self.fit_tfidfOri  = mat_tfidf
    
    def foundStrangeVoc(self):
        # # Strange Vocabulary 
        pKeys     = self.vocabulary.keys()
        elemnts   = self.tfidfOri.get_feature_names()
        ind       = [el in pKeys for el in elemnts] 
        not_Found = [elemnts[ii] for ii in range(len(elemnts)) if not ind[ii]]

        # # Find IDF
        countvec = CountVectorizer(binary = True, strip_accents='unicode')
        ppp = countvec.fit_transform(self.titleOri)
        nt  = np.array(ppp.sum(axis=0))[0]
        IDF = [np.log(len(self.titleOri) / ii) for ii in nt]
        IFM = [np.log(1 + max(nt) / ii) for ii in nt]

        nf_NT  = [nt[countvec.vocabulary_[element]] for element in not_Found]
        nf_IDF = [IDF[countvec.vocabulary_[element]] for element in not_Found]
        nf_IFM = [IFM[countvec.vocabulary_[element]] for element in not_Found]

        # # max TF-IDF
        tfidf_max = self.fit_tfidfOri.max(axis = 0).toarray()
        nf_TFIDF  = [tfidf_max[0][self.tfidfOri.vocabulary_[element]] for element in not_Found]
        resul = pd.DataFrame({'texto': not_Found, 'N_t' : nf_NT, 'IDF' : nf_IDF, 'IFM' : nf_IFM, 'Max_tf-IDF': nf_TFIDF})
        resul.sort_values(by = ['IDF', 'N_t'], ascending = [1, 0])        
        return resul
    
# # Funcion para extraer tokens
def parsetexts(tokens, tfs):
    index  = zip(tfs.nonzero()[0], tfs.nonzero()[1])
    tokens = [(key, [tokens[thing[1]] for thing in group]) for key, group in 
              groupby(index, lambda x: x[0])]
    return(tokens)

remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
cachedStopWords = stopwords.words("english")
    
def removeStops(text):
    # # Arreglando textos (minuzculas, quitando puntuaciones, asentos, palabras repetidas, duplicados)
    result = ' '.join([word.lower() for word in text.translate(remove_punctuation_map).split() 
                       if word not in cachedStopWords and not any([c.isdigit() for c in word])])
    return(result)

def makeTFidf(sin_puntuacion, max_df = 1.0, min_df=1):    
    exclude        = set(string.punctuation)
    sin_puntuacion = [removeStops(ww) for ww in sin_puntuacion]
    listDupli      = np.unique([re.sub("\s", "", ww) for ww in sin_puntuacion], return_index = True)
    #sin_duplicados = np.array(sin_puntuacion)[np.array(listDupli[1])]
    #sin_duplicados = sin_duplicados[np.where(sin_duplicados != u'')]
    sin_duplicados  = sin_puntuacion
    def stemText(text):
        return ' '.join([SnowballStemmer("english").stem(word) for word in text.split()])
    
    sin_duplicados = [stemText(ww) for ww in sin_duplicados]
    tfidf = TfidfVectorizer(stop_words = 'english', # tokenizar y eliminar stops word
                            strip_accents='unicode', max_df = max_df, min_df = min_df)      # haciendo representacion tf - if
    tfs   = tfidf.fit_transform(sin_duplicados)           
    texts = parsetexts(tfidf.get_feature_names(), tfs)               # Construyendo diccionario de terminos
    return([sin_duplicados, texts, tfidf, tfs])

def hisTFidf(tfs):
    rcParams['figure.figsize'] = 16, 4
    calDF_Ori = pd.DataFrame({'DF' : [sum(np.array(tfs.T.getrow(ww).toarray()[0]) != 0) / float(tfs.shape[0]) 
                                      for ww in range(tfs.shape[1])] , 
                  'N_t': [sum(np.array(tfs.T.getrow(ww).toarray()[0]) != 0) for ww in range(tfs.shape[1])]})
    print(calDF_Ori.describe())
    ptCorte = calDF_Ori.describe().loc['mean', 'DF'] + 3 * calDF_Ori.describe().loc['std', 'DF']
    ptCorte = ptCorte.round(5)
    print("Corte sugerido >: " + str(ptCorte))
    calDF_Ori.hist()
    return(calDF_Ori)

##########################################################################################################################
# # Word2Vec
##########################################################################################################################
ENGLISH_STOP_WORDS = [
    "a", "about", "above", "across", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done",
    "down", "due", "during", "each", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty", "fill",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "front", "full", "further", "get", "give", "go",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "in", "inc", "indeed",
    "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter",
    "latterly", "least", "less", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "move", "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "system", "take", "ten", "than", "that", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they", "thick", "thin",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us",
    "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
    "whence", "whenever", "where", "whereafter", "whereas", "whereby",
    "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself",
    "yourselves"]

#modelW2V = Word2Vec.load_word2vec_format('/home/jearevaloo/GoogleNews-vectors-negative300.bin', binary=True)
def makeWord2Vec(sin_puntuacion, modelW2V, tfidf = False, flagLower = False):        
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    cachedStopWords = stopwords.words("english")
    def removeStops(text):
        # # Arreglando textos
        if not flagLower:
            result = ' '.join([re.sub(u'\u2019', ' ', word) for word in text.translate(remove_punctuation_map).split() 
                               if word.lower() not in cachedStopWords and not any([c.isdigit() for c in word]) and 
                               word.lower() not in ENGLISH_STOP_WORDS])
        else:
            result = ' '.join([re.sub(u'\u2019', ' ', word.lower()) for word in text.translate(remove_punctuation_map).split() 
                               if word.lower() not in cachedStopWords and not any([c.isdigit() for c in word]) and 
                               word.lower() not in ENGLISH_STOP_WORDS])
        return(result.split())
    sin_puntuacion = [removeStops(ww) for ww in sin_puntuacion]    
    if tfidf:
        tfidf = TfidfVectorizer(strip_accents='unicode')
        tfs   = tfidf.fit_transform([' '.join(text) for text in sin_puntuacion])
    matrix = []; countNF = 0
    totalDoc = sum([len(ww) for ww in sin_puntuacion])
    for ii in range(len(sin_puntuacion)):        
        if ii % 100 == 0: print("Title No --" + str(ii) +  "--")
        vector  = [float(0)] * len(modelW2V[modelW2V.index2word[0]])
        sumWeig = 0
        for ww in sin_puntuacion[ii]:
            if ww in modelW2V.index2word:
                if tfidf:                    
                    if ww in tfidf.vocabulary_.keys():
                        isCol   = tfidf.vocabulary_[ww]                        
                        weigth = tfs.getrow(ii).getcol(isCol).toarray()[0][0]
                        vector += weigth * modelW2V[ww]
                        sumWeig += weigth
                else:
                    vector  += modelW2V[ww]
                    sumWeig += 1
            else:
                countNF += 1
        vector = np.divide(vector, sumWeig)
        matrix.append(vector)
    print("Termns not found :" + str(countNF) + " ("+ str(round(countNF/float(totalDoc) * 100, 2)) + ")")
    matrix = np.matrix(matrix)
    matrix = sparse.csr_matrix(matrix)
    return(matrix)

def semW2V(sin_puntuacion, modelW2V, flagLower = False):        
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    cachedStopWords = stopwords.words("english")
    def removeStops(text):
        # # Arreglando textos
        if not flagLower:
            result = ' '.join([re.sub(u'\u2019', ' ', word) for word in text.translate(remove_punctuation_map).split() 
                               if not any([c.isdigit() for c in word]) and word.lower() not in ENGLISH_STOP_WORDS])
        else:
            result = ' '.join([re.sub(u'\u2019', ' ', word.lower()) for word in text.translate(remove_punctuation_map).split() 
                               if not any([c.isdigit() for c in word]) and word.lower() not in ENGLISH_STOP_WORDS])
        return(result.split())
    sin_puntuacion = [removeStops(ww) for ww in sin_puntuacion]    
    tfidf_model = TfidfVectorizer(strip_accents='unicode')
    tfs   = tfidf_model.fit_transform([' '.join(text) for text in sin_puntuacion])
    auxDict = sorted(tfidf_model.vocabulary_.items())
    auxDict = [ww for ww, pos in auxDict if ww in modelW2V.index2word]
    indxAux = [tfidf_model.vocabulary_[ww] for ww in auxDict]
    LT      = modelW2V[auxDict] 
    G       = np.cov(LT)
    countNF = tfs.shape[1]- len(auxDict)
    print("Termns not found :" + str(countNF) + " ("+ str(round(countNF/float(tfs.shape[1]) * 100, 2)) + ")")
    tfs     = tfs[:, indxAux]
    return(tfs*G*tfs.T)