import multiprocessing as mp
import matplotlib as mpl
import numpy as np
import pandas as pd
import copy_reg, types, re
import traceback, logging
import os, pickle, collections, random, pymf_II, sys
from sklearn.decomposition import NMF
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN, SpectralClustering, SpectralClustering
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn.cluster import KMeans

score_funcs = [
    metrics.cluster.adjusted_rand_score,
    metrics.homogeneity_score,
    metrics.v_measure_score,
    metrics.adjusted_mutual_info_score,
    metrics.mutual_info_score,]

from scipy import sparse
from scipy.sparse import coo_matrix, vstack
from scipy.sparse import csr_matrix, dia_matrix
from scipy.sparse.linalg import norm
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import dendrogram, linkage
from Kernel_Kmeans import KernelKMeans

sys.path.append('/home/jmcarrascoo/SSOKMF/ssokmf_gpu/src/')
import tables
import theano.tensor as T
#from mindlab.latent_semantic.okmf_logistic import OKMFLogistic



# # Funciones iniciales
def ComputeCosine(matrix = None, pathScose = None):     
    if not pathScose is None:
        if os.path.exists(pathScose):
            simCosine = pickle.load(open(pathScose, 'rb'))
            print("--- Load cosine similarity\n")
        else:
            if matrix is None:
                sys.error("Se debe pasar un parametro 'matrix' para el calculo de la matriz de similitudes")
            simCosine = 1 - pairwise_distances(matrix, metric="cosine")
            simCosine = sparse.csr_matrix(simCosine)
            pickle.dump(simCosine, open(pathScose, 'wb'))
            print("--- Save cosine similarity\n")
    else:
        simCosine = 1 - pairwise_distances(matrix, metric="cosine")
        simCosine = sparse.csr_matrix(simCosine)
    return(simCosine)
        
def showMatrix(tfs, cosen_DTR, cosen_other = None, titleDTR = 'TCOR Representation', titleOther = None):    
    if cosen_other is None:
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
    else:
        f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey=False)
    ax1.matshow(tfs.toarray(), cmap=plt.cm.Blues)
    ax1.grid()
    ax1.set_title('Original tf-idf', fontsize=14, fontweight='bold')
    if cosen_DTR.__class__.__name__ != "" and cosen_DTR.__class__.__name__ != "ndarray":
        cosen_DTR = cosen_DTR.toarray()
    ax2.matshow(cosen_DTR, cmap=plt.cm.Blues)
    ax2.grid()
    ax2.set_title(titleDTR, fontsize=14, fontweight='bold')        
    if not cosen_other is None:
        ax3.matshow(cosen_other.toarray(), cmap=plt.cm.Blues)
        ax3.grid()
        ax3.set_title(titleOther, fontsize=14, fontweight='bold')
        
# # Funciones para clustering de textos
def cluster_texts_kmeansII(tfidf_model, n_clus=3):
    # # Usando la funcion del paquete Kmeans de sklearn
    km_cluster = KMeans(n_clusters=n_clus)
    km_cluster.fit(tfidf_model)
    clustering = collections.defaultdict(list)
    for idx, label in enumerate(km_cluster.labels_):
        clustering[label].append(idx)
    return clustering, km_cluster.labels_, km_cluster.cluster_centers_


def reevaluate_centers_sp(clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.array(vstack(clusters[k]).mean(axis = 0)[0])[0])
    return newmu

def cluster_texts_DBSCAN(sim_Mat, distriMat):
    clusters   = DBSCAN(min_samples=1).fit_predict(sim_Mat.toarray())
    clustering = collections.defaultdict(list)
    centros    = collections.defaultdict(list)
    for idx, label in enumerate(clusters):
        clustering[label].append(idx)
        centros[label].append(distriMat[idx])
    return clustering, clusters, reevaluate_centers_sp(centros)

def reevaluate_centers(clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu

def cluster_texts_III(tfidf_model, n_clus=3):
    # # Usando funcion de clustering de paquete Ntkl
    cluster_GAA = GAAClusterer(n_clus)
    clusters    = cluster_GAA.cluster(tfidf_model, True)
    clustering  = collections.defaultdict(list)
    centros     = collections.defaultdict(list)
    for idx, label in enumerate(clusters):
        clustering[label].append(idx)
        centros[label].append(tfidf_model[idx])
    return clustering, clusters, reevaluate_centers(centros)

def dis_sim(a,b): 
    if (sparse.issparse(a) or sparse.issparse(b)):
        a = sparse.csr_matrix(a)
        b = sparse.csr_matrix(b)
        return(norm(np.subtract(a, b)))
    else: 
        return(np.linalg.norm(np.subtract(a, b)))
    
def coef_qerror(tfs_matrix, centers, solution):
        total_distance = 0.0
        for i in range(tfs_matrix.shape[0]):
            total_distance += dis_sim(tfs_matrix[i], centers[solution[i]])
        return total_distance / len(solution)

def coef_davies_bouldin(tfs_matrix, centers, solution):
        # promedio intra cluster
        clusters  = collections.defaultdict(list)
        for x in range(tfs_matrix.shape[0]):
            clusters[solution[x]].append(dis_sim(tfs_matrix[x], centers[solution[x]]))
        cluster_averages = reevaluate_centers(clusters)
        davies_bouldin   = 0.0        
        # Iterando para Clusters i diferente j 
        for i in range(len(set(solution))):
            d_i = []
            for j in range(len(set(solution))):
                if j != i:
                    # calculate the distance between the two centroids of i and j
                    d_ij = dis_sim(centers[i], centers[j])
                    d_i.append((cluster_averages[i] + cluster_averages[j]) / d_ij)
            davies_bouldin += max(d_i)
        davies_bouldin = davies_bouldin / len(set(solution))
        return davies_bouldin

def coef_Silueta(tfs_matrix, solution):
    solution = np.array(solution)
    # Comprobacion inicial y numero de clusters
    if len(solution) != tfs_matrix.shape[0]:
        sys.exit("The dimension of the vector solution is wrong")
    # promedio intra cluster
    n = range(len(solution))
    A = [np.mean([dis_sim(tfs_matrix[i], tfs_matrix[j]) for j in n if solution[i] == solution[j] and i != j]) for i in n]
    
    # Media distancia a cluster mas cercano
    B = [np.min([np.mean([dis_sim(tfs_matrix[i], tfs_matrix[j]) for j in np.where(solution == cur_label)[0]]) 
         for cur_label in set(solution) if not cur_label == solution[i]]) for i in n]
    sil_samples = (np.array(B) - np.array(A)) / np.maximum(np.array(A), np.array(B))  
    return np.mean(np.nan_to_num(sil_samples))


# # Funciones para correr iterativamente cluster y graficar resultados
def runClusters(sim_matrix, distriMat, vec_Cluster, simulations, funCluster):
    coef_QUA = pd.DataFrame(columns = ['QError', 'Davies-Bouldin', 'Silhouette'])
    best_q1, best_q2, best_q3  = (1000.00, 1000.00, 1000.00)
    for i in vec_Cluster:
        num_clusters = i
        print "---- evaluado para num_clusters = ", i
        for j in range(simulations):
            # Creando los clusters
            try:
                if funCluster.func_name == 'cluster_texts_spectral':
                    clustering = funCluster(sim_Mat = sim_matrix, n_clus = num_clusters, distriMat = distriMat)             
                if funCluster.func_name == 'cluster_texts_kernelKmeans':
                    clustering = cluster_texts_kernelKmeans(tfidf_model = distriMat, n_clus = num_clusters)  
            except:
                next
                next
            print len(clustering[2])
            print len(clustering[1])
            
            # medidas de validez internas
            cul_qua1 = coef_qerror(distriMat, clustering[2], clustering[1])
            if cul_qua1 < best_q1:
                best_q1 = cul_qua1
                best_c1 = clustering                
            cul_qua2 = coef_davies_bouldin(distriMat, clustering[2], clustering[1])
            if cul_qua2 < best_q2:
                best_q2 = cul_qua2
                best_c2 = clustering                   
            cul_qua3 = metrics.silhouette_score(distriMat, clustering[1])
            if cul_qua3 < best_q3:
                best_q3 = cul_qua3
                best_c3 = clustering
        aux_QUA  = pd.DataFrame({'QError' : best_q1, 'Davies-Bouldin' : best_q2, 'Silhouette' : best_q3}, index=[i])
        coef_QUA = coef_QUA.append(aux_QUA)
    return(coef_QUA)

def grafResult(resulQUA):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
    id_1 = np.array(resulQUA[["Davies-Bouldin"]])
    id_2 = np.array(resulQUA[["QError"]])
    id_3 = np.array(resulQUA[["Silhouette"]])
    ax1.plot(id_1, label="Davies-Bouldin")
    ax1.grid()
    ax1.legend()
    ax2.plot(id_2, color = 'red', label="QError")
    ax2.plot(id_3, color = 'green', label="Silhouette")
    ax2.grid()
    ax2.legend()
    
def comResult(resulQUA, resulQUA_II):
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
    # # Info first model
    id_1 = np.array(resulQUA[["Davies-Bouldin"]])
    id_2 = np.array(resulQUA[["QError"]])
    id_3 = np.array(resulQUA[["Silhouette"]])
    # # Info second model
    id_1_II = np.array(resulQUA_II[["Davies-Bouldin"]])
    id_2_II = np.array(resulQUA_II[["QError"]])
    id_3_II = np.array(resulQUA_II[["Silhouette"]])
    # # Plot
    ax1.plot(id_1, label="Davies-Bouldin")
    ax1.plot(id_1_II, label="Davies-Bouldin_II", linestyle = "--")
    ax1.grid()
    ax1.legend()
    ax2.plot(id_2, color = 'red', label="QError")
    ax2.plot(id_2_II, color = 'red', label="QError_II", linestyle = "--")
    ax2.plot(id_3, color = 'green', label="Silhouette")
    ax2.plot(id_3_II, color = 'green', label="Silhouette_II", linestyle = "--")
    ax2.grid()
    ax2.legend()


score_funcs = [
    metrics.cluster.adjusted_rand_score,
    metrics.homogeneity_score,
    metrics.v_measure_score,
    metrics.adjusted_mutual_info_score,
    metrics.mutual_info_score]

#funcion nueva para validacion de clusters

class oneCluster:    
    def __init__(self, DTR, groups, cluster, centers, labels = None):
        # # Internal Evaluation
        self.coef_qerror = self.coef_qerror(groups, cluster, centers, DTR)
        self.coef_db     = self.coef_davies_bouldin(groups, cluster, centers, DTR)
        self.coef_silhou = metrics.silhouette_score(DTR, cluster)                       
        self.scoresExter = np.zeros((1, len(score_funcs) + 1))

        # # External Evaluation
        if (not labels is None):
            for mesure in range(len(score_funcs)):
                self.scoresExter[0, mesure] = score_funcs[mesure](labels, cluster)
            self.scoresExter[0, mesure + 1] = self.calculate_purity(groups, cluster, centers, labels)
            

    def dis_sim(a,b): 
        if (sparse.issparse(a) or sparse.issparse(b)):
            a = sparse.csr_matrix(a)
            b = sparse.csr_matrix(b)
            return(norm(np.subtract(a, b)))
        else: 
            return(np.linalg.norm(np.subtract(a, b)))
    
    def calculate_purity(self, groups, cluster, centers, labels):
        if labels is None:
            sys.exit("The dimension of the vector solution is wrong")        
        # find out what class is most frequent in each cluster        
        cluster_classes = {}
        for cluster_kk in groups.keys():
            # get the indices of rows in this cluster
            indices        = groups[cluster_kk]            
            cluster_labels = labels[indices]
            # assign the most common label to be the label for this cluster
            cluster_classes[cluster_kk] = cluster_labels.value_counts().idxmax()
        
        def get_label(cluster):
            """
            Get the label for a sample based on its cluster.
            """
            return cluster_classes[cluster]
        
        vfunc = np.vectorize(get_label)
        # get the list of labels as determined by each cluster's most frequent 
        # label
        labels_by_clustering = vfunc(cluster)

        # See how the clustering labels compare with the actual labels.  
        # Return the percentage of indices in agreement.
        num_agreed = 0
        for ind in range(len(labels_by_clustering)):
            if labels_by_clustering[ind] == labels[ind]:
                num_agreed += 1        
        return float(num_agreed) / labels_by_clustering.size
    
    def coef_qerror(self, groups, cluster, centers, DTR):
        total_distance = 0.0        
        for i in range(DTR.shape[0]):
            total_distance += dis_sim(DTR[i], centers[cluster[i]])
        return total_distance / len(cluster)

    def coef_davies_bouldin(self, groups, cluster, centers, DTR):
            # promedio intra cluster
            clusters  = collections.defaultdict(list)
            for x in range(DTR.shape[0]):
                clusters[cluster[x]].append(dis_sim(DTR[x], centers[cluster[x]]))
            cluster_averages = reevaluate_centers(clusters)
            davies_bouldin   = 0.0        
            # Iterando para Clusters i diferente j 
            for i in range(len(groups)):
                d_i = []
                for j in range(len(groups)):
                    if j != i:
                        # calculate the distance between the two centroids of i and j
                        d_ij = dis_sim(centers[i], centers[j])
                        d_i.append((cluster_averages[i] + cluster_averages[j]) / d_ij)
                davies_bouldin += max(d_i)
            davies_bouldin = davies_bouldin / len(set(cluster))
            return davies_bouldin

def cluster_texts_OKMF(myObject, n_clus = 3, labels = None):
    # parameter definiton for OKMF
    params = {
            'lambda1':0.001,
            'lambda2':0.001,
            'lambda3':0.001,
            'topics': 20,
            'alpha': 1,
            'beta': 1,
            'minibatch': 100,
            'epochs' : 20,
            'budget': 300,
            'kernel' : {'metric':'rbf', 'gamma':1e1},
            'learning' : {'rule':'rmsprop', 'gamma':0.1, 'lambda':1e-6, 'epsilon':0.01},
            'y_loss' : 'squared'
    }
    
    okmf = OKMFLogistic(**params)
    # start learning process (unsupervised)
    X = np.float32(myObject.DTR.toarray())
    okmf.fit(X, None)
    # projection to semantic space
    H = okmf.predict_h(X)
    # # Usando la funcion del paquete Kmeans de sklearn
    try:
        km_cluster = KMeans(n_clusters = n_clus)
        km_cluster.fit(H)
        clustering = collections.defaultdict(list)
        centros    = collections.defaultdict(list)
        for idx, label in enumerate(km_cluster.labels_):
            clustering[label].append(idx)
            centros[label].append(myObject.DTR[idx])
        clusters = oneCluster(DTR = myObject.DTR, groups = clustering, cluster = km_cluster.labels_, 
                              centers = reevaluate_centers_sp(centros),labels = labels)
        return clusters
    except Exception as e:
        print logging.error(traceback.format_exc())
        return None    

def cluster_texts_kernelKmeans(myObject, n_clus = 3, labels = None):
    # # Usando la funcion del paquete Kmeans de sklearn
    try:
        km_cluster = KernelKMeans(n_clusters = n_clus, max_iter=100, random_state=0, 
                                    verbose=1, kernel = 'cosine')
        km_cluster.fit(myObject.matrix)
        clustering = collections.defaultdict(list)
        centros    = collections.defaultdict(list)
        for idx, label in enumerate(km_cluster.labels_):
            clustering[label].append(idx)
            centros[label].append(myObject.DTR[idx])
        clusters = oneCluster(DTR = myObject.DTR, groups = clustering, cluster = km_cluster.labels_, 
                              centers = reevaluate_centers_sp(centros), labels = labels)
        return clusters
    except Exception as e:
        print logging.error(traceback.format_exc())
        return None    

def cluster_texts_spectral(myObject, n_clus = 3, labels = None):
    try:
        clusters   = SpectralClustering(n_clus, affinity = 'precomputed')
        clusters   = clusters.fit_predict(myObject.matrix.toarray())
        clustering = collections.defaultdict(list)
        centros    = collections.defaultdict(list)
        for idx, label in enumerate(clusters):
            clustering[label].append(idx)
            centros[label].append(myObject.DTR[idx])      
        clusters = oneCluster(DTR = myObject.DTR, groups = clustering, cluster = clusters, 
                                centers = reevaluate_centers_sp(centros), labels = labels)
        return clusters
    except Exception as e:
        print logging.error(traceback.format_exc())
        return None
    
def cluster_texts_NMF_sim(myObject, n_clus = 3, labels = None):
    try:
        nmf_cluster = NMF(n_components = n_clus, init='random', random_state=0)
        nmf_cluster.fit(myObject.matrix)
        V = nmf_cluster.components_
        #U = nmf_cluster.fit_transform(X = myObject.DTR)
        #V = V.T * np.sqrt(np.sum(U ** 2, axis = 1))
        #W = sklearn.preprocessing.normalize(W)
        clusters = V.argmax(axis=0)              
        clustering = collections.defaultdict(list)
        centros    = collections.defaultdict(list)
        for idx, label in enumerate(clusters):
            clustering[label].append(idx)
            centros[label].append(myObject.DTR[idx])
        clusters = oneCluster(DTR = myObject.DTR, groups = clustering, cluster = clusters, 
                              centers = reevaluate_centers_sp(centros), labels = labels)
        return clusters
    except Exception as e:
        print logging.error(traceback.format_exc())
        return None 

def cluster_texts_NMF(myObject, n_clus = 3, labels = None):
    try:
        nmf_cluster = NMF(n_components = n_clus, init='random', random_state=0)
        nmf_cluster.fit(myObject.DTR.T)
        V = nmf_cluster.components_
        #U = nmf_cluster.fit_transform(X = myObject.DTR)
        #V = V.T * np.sqrt(np.sum(U ** 2, axis = 1))
        #W = sklearn.preprocessing.normalize(W)
        clusters = V.argmax(axis=0)              
        clustering = collections.defaultdict(list)
        centros    = collections.defaultdict(list)
        for idx, label in enumerate(clusters):
            clustering[label].append(idx)
            centros[label].append(myObject.DTR[idx])
        clusters = oneCluster(DTR = myObject.DTR, groups = clustering, cluster = clusters, 
                              centers = reevaluate_centers_sp(centros), labels = labels)
        return clusters
    except Exception as e:
        print logging.error(traceback.format_exc())
        return None

def cluster_texts_CNMF(myObject, n_clus = 3, *args):
    try:    
        cnmf_cluster = CNMF(n_components = n_clus, num_bases=20)
        cnmf_cluster.factorize(show_progress=False, niter=100)
        H = nmf_cluster.components_
        W = nmf_cluster.fit_transform(X = tfidf_model)
        H = H * np.sqrt(np.sum(W ** 2, axis = 1))
        W = sklearn.preprocessing.normalize(W)
        clusters = H.argmax(axis=0)
        clustering = collections.defaultdict(list)
        centros    = collections.defaultdict(list)
        for idx, label in enumerate(clusters):
            clustering[label].append(idx)
            centros[label].append(distriMat[idx])
        return clustering, clusters, reevaluate_centers_sp(centros) 
    except:
        return None

class resulCluster:
    """Algoritmos de agrupacion"""
    def cluster_texts_spectral(self, n_clus = 3, labels = None):
        try:
            clusters   = SpectralClustering(n_clus, affinity = 'precomputed')
            clusters   = clusters.fit_predict(self.matrix.toarray())
            clustering = collections.defaultdict(list)
            centros    = collections.defaultdict(list)
            for idx, label in enumerate(clusters):
                clustering[label].append(idx)
                centros[label].append(self.DTR[idx])      
            clusters = oneCluster(DTR = self.DTR, groups = clustering, cluster = clusters, 
                                  centers = reevaluate_centers_sp(centros), labels = labels)
            return clusters
        except Exception as e:
            print logging.error(traceback.format_exc())
            return None



    """ Represents a results of Cluster"""
    def Evaluation(self, method, num_cluster, ntime, labels = None,  mProcessing = True):
        resulFinal = collections.defaultdict()
        for kClust in num_cluster:
            print("Procesando para " + str(kClust) + " grupos")
            if mProcessing:
                pool     = mp.Pool(processes = 4)
                results  = [pool.apply_async(method, args = (self, kClust, labels)) for x in range(0, ntime)]
                resulFinal[kClust] = [p.get() for p in results]
                pool.close()
                pool.join()
            else:
                for time in range(ntime):
                    if kClust not in resulFinal.keys():
                        resulFinal[kClust] = [method(self, kClust, labels)]
                    else:
                        resulFinal[kClust].append(method(self, kClust, labels))
        return(resulFinal)    
    
    def __init__(self, matrixSimilarity, DTR, fileCluster = None, listMethod = None,  **kwargs):
        if not fileCluster is None and (not os.path.exists(fileCluster)):
            self.matrix = matrixSimilarity   
            self.DTR    = DTR
            if listMethod is None:
                self.listMethod = [cluster_texts_spectral, cluster_texts_kernelKmeans]    
            else:
                self.listMethod = listMethod
            self.resulQUA = []
            for method in self.listMethod:
                print "- evaluado para el metodo = ", method.func_name            
                auxQUA = self.Evaluation(method = method, **kwargs)
                self.resulQUA.append(auxQUA)
            if not fileCluster is None:
                pickle.dump(self, open(fileCluster, "wb"))            
        else:
            print "--- Load cluster results from:" + fileCluster
            auxObject       = pickle.load(open(fileCluster, "rb")) 
            self.DTR        = auxObject.DTR
            self.listMethod = auxObject.listMethod
            self.matrix     = auxObject.matrix
            self.resulQUA   = auxObject.resulQUA
            
            
def getResul(resulClust, flagInternal = True):
    numClus   = resulClust.resulQUA[0].keys()
    numRepli  = len(resulClust.resulQUA[0][numClus[0]])
    numMethod = len(resulClust.listMethod)
    internal = np.zeros((numMethod, len(numClus), numRepli,  3))
    external = np.zeros((numMethod, len(numClus), numRepli, len(score_funcs) + 1))
    for ii in range(numMethod):    
        for jj, kk in enumerate(numClus):
            for zz in range(numRepli):
                resul = resulClust.resulQUA[ii][kk][zz]
                if not resul is None:
                    internal[ii, jj, zz, :] = [resul.coef_qerror, resul.coef_db, resul.coef_silhou]
                    external[ii, jj, zz, :] = resul.scoresExter
    if flagInternal:
        return(internal) 
    else:
        return(external) 

def plotResul(resulClust, title = "Clustering Results", flagInternal = True):
    matrixResul = np.array(getResul(resulClust, flagInternal= flagInternal))
    auxKclust   = resulClust.resulQUA[0].keys()
    if (np.sum(matrixResul) == 0):
        return None
    meanMesu = np.mean(matrixResul, axis = 2) 
    sdMesu   = np.std(matrixResul, axis = 2) 
    # # Etiquetas y graficos
    if flagInternal:
        labelMesure = ["Davies-Bouldin", "QError", "Silhouette"]
        f, axarr = plt.subplots(1, 3, sharey=False)
    else:
        labelMesure = [metrics.func_name for metrics in score_funcs]
        labelMesure.append('Purity')
        f, axarr = plt.subplots(1, 2, sharey=False)
        
    allStyle = ['solid', 'dashed', 'dashdot', 'dotted', '-', '--', '-.', ':']
    colStyle = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'white']
    for resul in range(matrixResul.shape[0]):                
        auxMethods = resulClust.listMethod[resul].func_name
        auxMethods = re.sub("cluster.texts.", "", auxMethods)
        for ii, mesure in enumerate(labelMesure):
            if flagInternal:
                indice = ii
                auxLabel =  "(" + auxMethods + ")"
                auxTile  =  mesure
                auxStyle = allStyle[resul]
            else:
                indice = resul
                auxLabel = "(" + mesure + ")"
                auxTile = auxMethods      
                auxStyle = 'solid'
            axarr[indice].plot(auxKclust, meanMesu[resul, :, ii], label = auxLabel, linestyle = auxStyle, color = colStyle[ii])
            axarr[indice].grid(); axarr[indice].legend(); axarr[indice].set_title(auxTile)
    f.suptitle(title, fontsize = 16)

def tabResum(resulMethods, labelDTR, nCluster = 3):
    labelMesure = ["Davies-Bouldin", "QError", "Silhouette"]
    labelMesure.extend([metrics.func_name for metrics in score_funcs])
    labelMesure.append('Purity')
    resulPanda = pd.DataFrame()
    for ii, method in enumerate(resulMethods):
        auxKclust = np.where(np.array(method.resulQUA[0].keys()) == nCluster)[0]
        auxMethods = [re.sub("cluster.texts.", "", me.func_name).capitalize() for me in method.listMethod]
        auxMethods = [me + "(" + labelDTR[ii] + ")" for me in auxMethods]
        matrixResulI = np.array(getResul(method, flagInternal= True))[:, auxKclust[0], :, :]
        matrixResulE = np.array(getResul(method, flagInternal= False))[:, auxKclust[0], :, :]
        matrixResulI = np.mean(matrixResulI, axis = 1)
        matrixResulE = np.mean(matrixResulE, axis = 1)
        resulFinal   = np.hstack((matrixResulI, matrixResulE))
        resulFinal   = pd.DataFrame(resulFinal.T, columns = auxMethods, index = labelMesure)
        resulPanda   = pd.concat([resulPanda, resulFinal], axis=1)
    return(resulPanda)