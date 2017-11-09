import requests, inspect
import pandas as pd
import pickle, os, re
import sPickle
import json
import numpy as np
import nltk, collections, re, string, scipy, gc, os
from urllib import quote_plus
from scipy.sparse import csr_matrix, dia_matrix


class scopusResult:
    """Representa una consulta de scopus."""
    def __init__(self):
        self.scopID = None # Electronic ID from SCOPUS
        self.tile = None   # The title of article
        self.link = None   # The external link
        self.abstract = None  # The description of the link
        self.keywords = None  # Thumbnail link of website (NOT implemented yet)
        self.authors = None  # Cached version link of page (NOT implemented yet)
    
    def __repr__(self):
        name = self._limit_str_size(self.tile, 55)
        abst = self._limit_str_size(self.abstract, 49)
        list_scopus = ["scopusResult(",
                       "name={}".format(name), "\n", " " * 13,
                       "description={}".format(abst)]
        return "".join(list_scopus)
    def _limit_str_size(self, str_element, size_limit):
        if not str_element:
            return None
        elif len(str_element) > size_limit:
            return str_element[:size_limit] + ".."
        else:
            return str_element

class searchScopus:    
    def __init__(self, query, elseiverKey, fileName, basePath = '../scopus-data/', refresh = False, numAbs = 30, flagKEY = False, 
                 flagALL = True):
        self.flagKEY = flagKEY
        self.flagALL = flagALL
        self.fileName = fileName
        self.ApyKey   = elseiverKey # API-KEY Elseiver API                
        self.base     = basePath
        self.refresh  = refresh            
        self.response = self._get_search_url(query, numAbs) # Url with options
        self.url      = self.response[1]                         
        self.get_abstract()
        
    def _normalize_query(self, query, flagAND = True):
        if re.match("SUBJAREA\\(\\w+\\)", query) or self.flagALL:
            return query.strip().replace(":", "%3A").replace("+", "%2B").replace("&", "%26").replace(" ", "+")
        else:
            if not self.flagKEY:
                if flagAND:
                    return "TITLE-ABS-KEY%28" + query.strip().replace(":", "%3A").replace("+", "%2B").replace("&", "%26").replace(" ", "+") + "%29"
                else:
                    return "TITLE-ABS-KEY%28" + '+OR+'.join(query.strip().replace(":", "%3A").replace("+", "%2B").replace("&", "%26").split()) + "%29"        
            else:
                return "AUTHKEY%28" + query.strip().replace(":", "%3A").replace("+", "%2B").replace("&", "%26").replace(" ", "+") + "%29"

    def _get_search_url(self, query, num_resul = 3, view = 'COMPLETE'):
        # query:     String con cadena original
        # num_resul: Numeros de resultados esperados
        # key:       Key of API Scopus
        # view:      This alias represents the list of elements that will be returned in the response
        
        # # Cargar archivo si existe
        
        if not os.path.exists(self.base):
            os.makedirs(self.base)
        fname = '{0}/{1}'.format(self.base, self.fileName) + '.json'
        if os.path.exists(fname) and not self.refresh:
            with open(fname, 'r') as input:
                auxJson = json.load(input)                
        else:        
            proxies = {"http": "http://jmcarrascoo:jorge2013@proxyapp.unal.edu.co:8080/",
                       "https": "https://jmcarrascoo:jorge2013@proxyapp.unal.edu.co:8080/"}
            cons_parm = {'apikey':self.ApyKey, 'view': view, 
                         'count':num_resul, 
                         'start':1,
                         'facets': 'language(count=1)'}
            headers   = {'User-Agent': 'Chrome/45.0.2454.85', 
                         'X-ELS-APIKey': self.ApyKey, 'Accept':'application/json'}
            auxQuery  = "http://api.elsevier.com/content/search/scopus?query=%s" %(self._normalize_query(query))
            response  = requests.get(auxQuery, params = cons_parm, proxies=proxies)
            auxJson   = response.json()            
            auxJson['url_mia'] = response.url
            print(response.url)
            indEmpty  = response.json()[u'search-results'][u'opensearch:totalResults'] == u'0'
            if (u'error' in auxJson[u'search-results']['entry'][0].keys() or indEmpty):
                auxQuery = "http://api.elsevier.com/content/search/scopus?query=%s" %(self._normalize_query(query, flagAND = False))
                response = requests.get(auxQuery, params = cons_parm, proxies=proxies)
                auxJson  = response.json()
                auxJson['url_mia'] = response.url
            print(response.json()[u'search-results'][u'opensearch:totalResults'])   
            # # Guardando archivo        
            with open(fname, 'w') as outfile:
                json.dump(auxJson, outfile) 
        auxUrl = auxJson['url_mia']
        return (auxJson, auxUrl)
        
    def get_abstract(self):     
            # # Haciendo Consulta
            results   = self.response[0]
            # # capturando informacion
            busqueda = []
            if (not u'error' in results[u'search-results']['entry'][0].keys()):
                for articulo in results[u'search-results']['entry']:                                        
                    res  = scopusResult()
                    res.scopID = articulo['eid']
                    if u'dc:title'in articulo.keys():
                        res.tile = articulo[u'dc:title']                        
                    res.link = articulo[u'prism:url']
                    if u'dc:description'in articulo.keys():
                        res.abstract = articulo[u'dc:description']                       
                    else:
                        continue
                    if u'authkeywords' in articulo.keys():
                        res.keywords = articulo[u'authkeywords'] 
                    if u'author' in articulo.keys():
                        if isinstance(articulo[u'author'], dict):
                            res.authors = articulo[u'author'][u'given-name'] + " " +  articulo[u'author'][u'surname']
                        else:                                                           
                            res.authors = [ww[u'given-name'] + " " +  str(ww[u'surname']) if isinstance(ww[u'surname'], (bool)) else ww[u'surname']
                                           for ww in articulo[u'author'] if (not ww[u'given-name'] is None) and (not ww[u'surname'] is None)]
                    busqueda.append(res)
                self.response = busqueda            
            else:
                self.response = []

class searchSciense(searchScopus):
    def _normalize_query(self, query, flagAND = True):
        if flagAND:
            return query.strip().replace(":", "%3A").replace("+", "%2B").replace("&", "%26").replace(" ", "+")
        else:
            return '+OR+'.join(query.strip().replace(":", "%3A").replace("+", "%2B").replace("&", "%26").split())
    
    def _get_search_url(self, query, num_resul = 3, view = 'COMPLETE'):
        # query:     String con cadena original
        # num_resul: Numeros de resultados esperados
        # key:       Key of API ciense
        # view:      This alias represents the list of elements that will be returned in the response        
        # # Cargar archivo si existe
        base     = '../ciense-data/'
        if not os.path.exists(base):
            os.makedirs(base)
        fname = '{0}/{1}'.format(base, self.fileName) + '.json'
        if os.path.exists(fname) and not self.refresh:
            with open(fname, 'rb') as input:
                auxJson = json.load(input)
        else:
            proxies = {"http": "http://jmcarrascoo:jorge2013@proxyapp.unal.edu.co:8080/",
                       "https": "https://jmcarrascoo:jorge2013@proxyapp.unal.edu.co:8080/"}
            cons_parm = {'apikey':self.ApyKey, 'view': view,
                         'start': 0, 'count':num_resul, 'content': 'all', 
                         'facets': 'language(count=1)'}
            headers   = {'User-Agent': 'Chrome/45.0.2454.85', 
                         'X-ELS-APIKey': self.ApyKey, 'Accept':'application/json'}
            auxQuery  = "http://api.elsevier.com:80/content/search/scidir?facets=language(count=1)&query=%s" %(self._normalize_query(query))
            response  = requests.get(auxQuery, params=cons_parm, proxies=proxies)
            auxJson   = response.json()
            auxJson['url_mia']  = response.url
            print(response.url)
            if (u'error' in auxJson[u'search-results']['entry'][0].keys() or '504 Gateway Time-out' in response.text):
                auxQuery = "http://api.elsevier.com:80/content/search/scidir?facets=language(count=1)&query=%s" %(self._normalize_query(query, flagAND = False))                
                response = requests.get(auxQuery, params = cons_parm, proxies=proxies)                
                if not '504 Gateway Time-out' in response.text:
                    auxJson  = response.json()
                    auxJson['url_mia'] = response.url
                    
            # # Guardando archivo        
            with open(fname, 'w') as outfile:
                json.dump(auxJson, outfile)            
        
        auxUrl = auxJson['url_mia']
        return (auxJson, auxUrl)
    
    #def __init__(self, **kwargs):
        #searchScopus.__init__(self, **kwargs) 

def buscarList(listQuery, basePath, flagKEY, fileOut = None, verbose = True, flagALL = True):       
    if (len(listQuery) < 150):
        numMod = 10
    else:
        numMod = 100
    if verbose:
        print '---- Consultas en scopus para:'
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        for i in args:
            if i == 'listQuery':
                values[i] = type(values[i])
            print "    %s = %s" % (i, values[i])
        print('    Largo =' + str(len(listQuery)))
        print('---------------------------')
    resultBus = collections.defaultdict(list)
    num = 0
    for query in listQuery.keys():
        num += 1
        if (listQuery[query] != ""):        
            #results = searchSciense(listQuery[query], elseiverKey = '3f8eebe2fd170110dc0c74a072238d9f', 
            results = searchScopus(listQuery[query], elseiverKey = '3f8eebe2fd170110dc0c74a072238d9f', 
                                   fileName = str(query), numAbs = 100, basePath = basePath, 
                                   flagALL = flagALL, flagKEY = flagKEY)
            resultBus[query] =results
            if num % numMod == 0: print("query No --" + str(num) +  "--")
    if not fileOut is None:
        # # Guardando vector de resultados Scopus
        sPickle.s_dump(resultBus.iteritems(), open(fileOut, 'w'))
    return(resultBus)

def orgDocum(resulList, listCate, numResul = None, flagUniq = True):
    catDoc, allDoc, allID = [], [], []
    for idDic, element in resulList.iteritems():
        exq       = element.response
        documents = [(ww.scopID, ww.tile, re.sub("(\xa9)?\\s+\\d{4}\\s+Elsevier Ltd.\\s?", "", ww.abstract), ww.keywords)
                     for ww in exq[0:numResul] if (not ww.abstract is None) and (not ww.keywords is None) and 
                     (not ww.scopID in allID)]        
        if flagUniq:
            allID.extend([pp.scopID for pp in exq])
        allDoc.extend(documents)
        catDoc.extend([listCate[idDic]] * len(documents))
    
    # # Armando data.frame
    datArti = pd.DataFrame(allDoc, columns=['ID_Scopus', 'Titulos', 'Abstract', 'Key_Words'])
    catDoc  = pd.DataFrame({'Categoria' : catDoc})
    datArti = pd.concat([catDoc, datArti], axis=1)
    print('Total Documentos :' + str(len(allDoc)))
    return(datArti)
               

# # Files with keyWords
def getResultSE(file): 
    auxUrl = []
    for element in sPickle.s_load(open(file)):
        auxUrl.append(element)
    return(auxUrl)
    
    
    
# # Comprobando que las url sean las de la lista indexQuery (Scopus Check)
# import pickle, collections, sPickle
# def _normalize_query(query, flagAND = True):
#         if flagAND:
#             return "TITLE-ABS-KEY(" + query.strip().replace(":", "%3A").replace("+", "%2B").replace("&", "%26").replace(" ", "+") + ")"
#         else:
#             return "TITLE-ABS-KEY(" + '+OR+'.join(query.strip().replace(":", "%3A").replace("+", "%2B").replace("&", "%26").split()) + ")"
# auxUrl = []
# for element in sPickle.s_load(open("Output/" + 'query_data_sc.spkl')):
#     auxUrl.append(element.url)
# auxUrl = [urllib.unquote(ww).encode('latin1').decode('utf-8') for ww in auxUrl]

# aux1 = []
# aux2 = []
# aux3 = []
# aux4 = []

# for query in indexQuery.keys():
#     if indexQuery[query] != '':
#         aux1.append("http://api.elsevier.com:80/content/search/scopus?start=0&count=100&query=%s" %(_normalize_query(indexQuery[query])) + "&apikey=3f8eebe2fd170110dc0c74a072238d9f&view=COMPLETE&facets=language(count=1)")
#         aux2.append("http://api.elsevier.com:80/content/search/scopus?start=0&count=100&query=%s" %(_normalize_query(indexQuery[query], flagAND = False)) + "&apikey=3f8eebe2fd170110dc0c74a072238d9f&view=COMPLETE&facets=language(count=1)")  
#         aux3.append("http://api.elsevier.com/content/search/scopus?query=%s" %(_normalize_query(indexQuery[query], flagAND = False)) + "&count=100&facets=language(count=1)&apikey=3f8eebe2fd170110dc0c74a072238d9f&view=COMPLETE")
#         aux4.append("http://api.elsevier.com/content/search/scopus?query=%s" %(_normalize_query(indexQuery[query])) + "&count=100&facets=language(count=1)&apikey=3f8eebe2fd170110dc0c74a072238d9f&view=COMPLETE")
        
# auxBol = [ww for ww in range(len(auxUrl)) if not (aux2[ww] == auxUrl[ww] or aux1[ww] == auxUrl[ww] or aux3[ww] == auxUrl[ww] or aux4[ww] == auxUrl[ww])]
# auxBol
        
# Eliminar articulos duplicado
#prueba = json.load(open("DB_keywords/18.json", "rb"))
#np.where([articulo['eid'] == u'2-s2.0-84954577939' for articulo in prueba[u'search-results']['entry']] )
#prueba[u'search-results']['entry'] = prueba[u'search-results']['entry'][0:7] + prueba[u'search-results']['entry'][8:97]
#json.dump(prueba, open("DB_keywords/18.json", "wb"))

#[ii for ii in range(len(sin_puntuacion)) if not ii in np.array(listDupli[1])]
#[ww for ww in range(1662) if sin_puntuacion[ww] == sin_puntuacion[1537]]


# # Find a Title
#ind = [ww == 8 for ww in datKeyW.loc[:, 'Categoria']]  
#ind2 = [ww == 'Generation of transgenic mouse model using PTTG as an oncogene' for ww in datKeyW.loc[:, 'Titulos']] 
#ind = [ind[ww] and ind2[ww] for ww in range(len(ind))]
#datKeyW.loc[ind,:][['Categoria', 'Titulos']]