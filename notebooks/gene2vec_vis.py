import io, os, sys, time, datetime, math, random

# dataproc
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt, matplotlib.image as img

# gene2vec
import gensim
from gensim.models.keyedvectors import KeyedVectors
from sklearn.decomposition import PCA
from multiprocessing import Pool

## word2vec parameters
dimension = 256     # dimension of the embedding
num_workers = 16    # number of worker threads
sg = 1              # sg =1, skip-gram, sg =0, CBOW
max_iter = 10       # number of iterations
window_size = 1     # The maximum distance between the gene and predicted gene within a gene list
negative = 3        # number of negative samples
txtOutput = True

# Environment variables
MSIGDB          = '../data/gene2vec/0_raw/msigdb.v6.2.symbols.gmt'
TCGA            = '../data/gene2vec/0_raw/processed_somatic_mutations_subset2'
INTOGEN         = '../data/gene2vec/0_raw/intogen-drivers-data.tsv'
GENE_PAIRS      = '../data/gene2vec/1_processed/gene_pairs_set.pkl'
GENE2VEC_DIR    = '../models/gene2vec/'
EXP1_DIR        = '../models/sklearn-ensemble/'
FILENAME        = 'gene2vec_dim_{0:d}_iter_'.format(dimension)
# load model and train on shuffled gene pairs
current_iter = 2
filename = GENE2VEC_DIR + FILENAME + str(current_iter)

def load_embeddings(file_name):
    model = KeyedVectors.load(file_name)
    wordVector = model.wv
    vocabulary, wv = zip(*[[word, wordVector[word]] for word, vocab_obj in wordVector.vocab.items()])
    return np.asarray(wv), vocabulary

# load test file
wv, vocabulary = load_embeddings(filename)

# shuffle index of genes
indexes = list(range(len(wv)))
random.shuffle(indexes)

topN = len(wv) # for now select all genes
rdWV = wv[indexes][:topN][:]
rdVB = np.array(vocabulary)[indexes][:topN]

print(wv.shape, rdWV.shape, rdVB.shape)

def plot_2d(data, title):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('dimension 1', fontsize = 15)
    ax.set_ylabel('dimension 2', fontsize = 15)
    ax.set_title(title, fontsize = 20)
    
    ax.scatter(data[:,0], data[:,1], s = 5, alpha=0.5)
    ax.grid()

# 50 component PCA for tSNE
pca = PCA(n_components=50)
pca.fit(rdWV)
pca_rdWV=pca.transform(rdWV)

n_iters = [100,500,1000]
# n_iters = [100,500,1000,10000,100000]

def tsne_worker(n_iter):
    """
    TSNE Worker, will run for CPU or GPU.
    """
    tsne = TSNE(n_components=2, 
                perplexity=30, 
                n_iter=n_iter, 
                learning_rate=200, 
                n_jobs=8)
    
    print('n_iter = {0:d} started'.format(n_iter))
    data = tsne.fit_transform(pca_rdWV)
    print('n_iter = {0:d} finished'.format(n_iter))
    return data, n_iter

import torch
cuda = torch.cuda.is_available()
if cuda:
    print("CUDA available, using GPU TSNE")
    from tsnecuda import TSNE
    for n_iter in n_iters:
        tsne = TSNE(n_components=2, 
                perplexity=30, 
                n_iter=n_iter, 
                learning_rate=200)
        data = tsne.fit_transform(pca_rdWV[200:, :])
        plot_2d(data, '2D tSNE, n_iter={0:d}'.format(n_iter))    
else:
    print("CUDA not available, using multi-core TSNE")
    from MulticoreTSNE import MulticoreTSNE as TSNE
    
    p = Pool(4)

    # generate tsne of different iteration in parallel
    results = p.map(tsne_worker, [100, 500])

    # plot tSNE
    for data, n_iter in results:
        plot_2d(data, '2D tSNE, n_iter={0:d}'.format(n_iter))