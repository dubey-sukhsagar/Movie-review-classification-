#!/usr/bin/env python
# coding: utf-8

# '''Unsupervised Sentiment Analysis:
# a) Write a code to extract TF-IDF features for each word from this dataset (remove
# the labels which are the last entry of each line). Use the average the TF-IDF feature
# as a document embedding vector (one feature per review) .
# b) Perform PCA on embedding vector to reduce to 10 dimensions.
# c) Train a two mixture diagonal covariance GMM on this data. Show the progress of
# the EM algorithm by coloring each data point by assigning the data point to the
# argmax of posterior probability of mixture component given the data point. Use the
# first two PCA dimensions for this scatter plot.
# d) Check if the cluster identity of (max posterior probability of each review) correlates
# with true label given for each review.'''

# In[132]:


#import Module
import os
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.feature_extraction.text import TfidfVectorizer
import math
#from scipy.sparse import csr_matrix
from collections import Counter
from tqdm import tqdm
import operator
from sklearn.preprocessing import normalize
import nltk
import random
import warnings
warnings.filterwarnings('ignore')


# In[133]:


#Reading file
path='C:\\Users\\Dubey\\Desktop\\IIsc_class_2\\E9 205_MLSP\\Assignment\\2'
os.chdir(path)
file=open(r'movieReviews1000.txt')
file=file.read()
print(type(file))


# In[136]:


#Create Corpus
x=[]
y=[]
CoList = file.split("\n")
for ele in CoList:
    x.append(ele[:-1])
    y.append(ele[-1])
#print(CoList)
corpus=x
print(corpus)


# In[137]:


#-------------Get IDF value 
def IDF(corpus, unique_words):
    idf_dict={}
    N=len(corpus)
    for i in unique_words:
        count=0
        for sen in corpus:
            if i in sen.split():
                count=count+1
                idf_dict[i]=(math.log((1+N)/(count+1)))+1
    return idf_dict 

def fit(whole_data):
    unique_words = []
    if isinstance(whole_data, (list,)):
        for x in whole_data:
            for y in x.split():
                if len(y)<2:
                    continue
                unique_words.append(y)
                unique_words = sorted(list(unique_words))
    
    Idf_values_of_all_unique_words=IDF(whole_data,unique_words)
    
    return list(set(unique_words)), Idf_values_of_all_unique_words

Vocabulary, idf_of_vocabulary=fit(corpus)


# In[138]:


# -----Remove English stopwords from Vocabulary 
#import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
stop = list(stopwords.words('english'))
for i in range(len(stop)):
    if stop[i] in Vocabulary:
        Vocabulary.remove(stop[i])
        
#print(list(Vocabulary.keys()))
#print(Vocabulary)
#print(Vocabulary.index('very'))
print(len(Vocabulary))        


# In[139]:


#Get Input for Model
def transform(dataset,vocabulary,idf_values):
    sparse_matrix= csr_matrix( (len(dataset), len(vocabulary)), dtype=np.float64)
    for row  in range(len(dataset)):
        number_of_words_in_sentence=Counter(dataset[row].split())
        for word in dataset[row].split():
            if word in  list(vocabulary):
                tf_idf_value=(number_of_words_in_sentence[word]/len(dataset[row].split()))*(idf_values[word])
                sparse_matrix[row,vocabulary.index(word)]=tf_idf_value
    
    output =normalize(sparse_matrix, norm='l2', axis=1, copy=True, return_norm=False)
    return output
    

final_output=transform(corpus,Vocabulary,idf_of_vocabulary)
print(final_output.shape) 


# In[140]:


# out of 3350 feature convert to 2 feature by PCA 
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
Xpca = pca.fit_transform(final_output.toarray())
print(Xpca)


# In[141]:


# Plot PCA DATA 
def plot_data(Xpca):
    x1=[]
    x2=[]
    y1=[]
    y2=[]
    for i in range(len(Xpca)):
        if y[i]=='1':
            x1.append(Xpca[i][0])
            y1.append(Xpca[i][1])
        else:
            x2.append(Xpca[i][0])
            y2.append(Xpca[i][1])
    
    plt.scatter(x1,y1, c='r', marker="o")
    plt.scatter(x2,y2, c='g', marker="o")
    plt.show()
    
plot_data(Xpca)    


# '''Train GMM as below with initialize with kmeans algo-
# E-step: To get Expectation calcualte r (responsibility) matix: 
# 
# ![estep.png](attachment:estep.png)
# 
# Pdf is given as below -
# ![estep1.png](attachment:estep1.png)
# M-step: Update mean, covariance and alpha(Pi) as below-
# 
# ![estep2.png](attachment:estep2.png)
# 
# '''

# In[148]:


# Gmm class which initialized with kmeans,with full covariance matrix and plot of each iteration 
class GMM:
    def __init__(self, n_components, max_iter , comp_names=None):
        
        self.n_componets = n_components
        self.max_iter = max_iter
        
        if comp_names == None:
            self.comp_names = [f"comp{index}" for index in range(self.n_componets)]
        else:
            self.comp_names = comp_names
        
    def mult_normal(self, X, mean_vector, covariance_matrix):
        s=(2*np.pi)**(-len(X)/2)*np.linalg.det(covariance_matrix)**(-1/2)*np.exp(-np.dot(np.dot((X-mean_vector).T, np.linalg.inv(covariance_matrix)), (X-mean_vector))/2)
        #print(np.exp(-np.dot(np.dot((X-mean_vector).T, np.linalg.inv(covariance_matrix)), (X-mean_vector))/2))
        #print('------')
        #print(np.linalg.det(covariance_matrix)**(-1/2))      

        return (s) 
    
    def kmeans(self,X):
        m=X.shape[0]
        n=X.shape[1]
        # creating an empty centroid array
        centroids=np.array([]).reshape(n,0)
        # creating k random centroids
        for k in range(self.n_componets):
            centroids=np.c_[centroids,X[random.randint(0,m-1)]]
        #print(centroids)
        for i in range(self.max_iter ):
            euclid=np.array([]).reshape(m,0)
            for k in range(self.n_componets):
                dist=np.sum((X-centroids[:,k])**2,axis=1)
                euclid=np.c_[euclid,dist]
            C=np.argmin(euclid,axis=1)+1
            cent={}
            for k in range(self.n_componets):
                cent[k+1]=np.array([]).reshape(n,0)
            for k in range(m):
                cent[C[k]]=np.c_[cent[C[k]],X[k]]
            for k in range(self.n_componets):
                cent[k+1]=cent[k+1].T
            for k in range(self.n_componets):
                centroids[:,k]=np.mean(cent[k+1],axis=0)
        return(centroids.T,cent)

    def fit(self, X):
        # By K_mans algo Spliting the data in n_componets
        # Initial computation of the mean-vector and covarience matrix
        
        mean_vector, new_X = self.kmeans(X)
        self.mean_vector=mean_vector
        self.covariance_matrixes = [(1/len(new_X))*np.dot((new_X[i]-self.mean_vector[i-1]).T, (new_X[i]-self.mean_vector[i-1])) for i in range(1,self.n_componets+1)]
        
        #Make covariance matrix diagonal 
        
        #self.covariance_matrixes=[np.diag(np.diag(self.covariance_matrixes[i])) for i in range(len(self.covariance_matrixes))] 
        # pi list contains the fraction of the dataset for every cluster
        self.pi = [len(new_X[i])/len(X) for i in range(1,self.n_componets+1)]
        
        # Deleting the new_X matrix becauseit not requied anymore
        del new_X
        
        '''print(self.pi)
        print(self.mean_vector)
        '''
        print('Initial Mean')
        print(self.mean_vector)
        print('Initial Covariance')
        print(self.covariance_matrixes)

        for iteration in range(self.max_iter):
            
            #plot 
            
            self.plot_2d_plot(X)

            ''' ----------------   E - STEP   ------------------ '''
            # Initiating the r matrix, evrey row contains the probabilities
            # for every cluster for this row
            self.r = np.zeros((len(X), self.n_componets))
            # Calculating the r matrix
            for n in range(len(X)):
                for k in range(self.n_componets):
                    self.r[n][k] = self.pi[k] * self.mult_normal(X[n], self.mean_vector[k], self.covariance_matrixes[k])
                    self.r[n][k] /= sum([self.pi[j]*self.mult_normal(X[n], self.mean_vector[j], self.covariance_matrixes[j]) for j in range(self.n_componets)])
                   # print(self.pi[k] * self.mult_normal(X[n], self.mean_vector[k], self.covariance_matrixes[k]))
            # Calculating the N
            N = np.sum(self.r, axis=0)
            #print(self.r)
            print('------{} Iteration-------'.format(iteration))


            ''' ---------------   M - STEP   --------------- '''
            # Initializing the mean vector as a zero vector
            self.mean_vector = np.zeros((self.n_componets, len(X[0])))
            # Updating the mean vector
            for n in range(len(X)):
                for k in range(self.n_componets):
                    self.mean_vector[k] += self.r[n][k] * X[n]
                    self.mean_vector = [1/N[k]*self.mean_vector[k] for k in range(self.n_componets)]
            
            #print(self.mean_vector)
            # Initiating the list of the covariance matrixes
            self.covariance_matrixes = [np.zeros((len(X[0]), len(X[0]))) for k in range(self.n_componets)]
            # Updating the covariance matrices
            for k in range(self.n_componets):
                 for n in range(len(X)):
                        self.covariance_matrixes[k] +=(self.r[n][k])*np.outer((X[n]-self.mean_vector[k]).T, (X[n]-self.mean_vector[k])) 
            self.covariance_matrixes = [1/N[k]*self.covariance_matrixes[k] for k in range(self.n_componets)]
            #self.covariance_matrixes=[np.diag(np.diag(self.covariance_matrixes[i])) for i in range(len(self.covariance_matrixes))] 
            #print(self.covariance_matrixes)
            # Updating the pi list
            self.pi = [N[k]/len(X) for k in range(self.n_componets)]
            #print(self.pi)
            
           
    def plot_2d_plot(self,X):
        Xpca=X
        x,k = np.meshgrid(np.sort(Xpca[:,0]),np.sort(Xpca[:,1]))
        XY = np.array([x.flatten(),k.flatten()]).T
        # Plot   
        fig = plt.figure(figsize=(10,10))
        ax0 = fig.add_subplot(111)
        
        ax0.scatter(Xpca[:,0],Xpca[:,1])
        for m,c in zip(self.mean_vector,self.covariance_matrixes):
            multi_normal = multivariate_normal(mean=m,cov=c)
            ax0.contour(np.sort(Xpca[:,0]),np.sort(Xpca[:,1]),multi_normal.pdf(XY).reshape(len(Xpca),len(Xpca)),colors='black',alpha=0.3)
            ax0.scatter(m[0],m[1],c='red',zorder=10,s=100)
        plt.show()

    def predict(self, X):
        probas = []
        for n in range(len(X)):
            probas.append([self.mult_normal(X[n], self.mean_vector[k], self.covariance_matrixes[k])
                           for k in range(self.n_componets)])
        cluster = []
        for proba in probas:
            cluster.append(self.comp_names[proba.index(max(proba))])
        return cluster       


# In[149]:


np.random.seed(0)
k=2
max_iter=10
gmm = GMM(k,max_iter)
gmm.fit(Xpca)


# In[ ]:





# In[ ]:




