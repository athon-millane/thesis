import io, os, sys, time, datetime, math, random

import pandas as pd, numpy as np, matplotlib.pyplot as plt, matplotlib.image as img
import gensim
from gensim.models.keyedvectors import KeyedVectors
from sklearn.decomposition import PCA
from multiprocessing import Pool

from .config import GENE2VEC, FIGURES


def load_embeddings(file):
    """
    Load trained w2v model, shuffle and return vectors and vocab array.
    """
    model = KeyedVectors.load(file)
    wordVector = model.wv
    vocab, wv = zip(*[[word, wordVector[word]] for word, vocab_obj in wordVector.vocab.items()])
    
    # load test file
    wv, vocab = np.asarray(wv), np.asarray(vocab)

    # shuffle index of genes
    indices = list(range(len(wv)))
    random.shuffle(indices)
    
    wv, vocab = wv[indices,:], vocab[indices]
    return wv, vocab


def gene_intersection(df1, df2):
    """
    Get intersection of genes (features) from df1 and df2.
    """
    gene_intersection = []
    for gene in df1:
        if gene in df2.columns.tolist():
            gene_intersection.append(gene)

    return gene_intersection

    
def embed_gene_vectors(data_df, gene_df):
    """
    Matrix multiply somatic mutation data by gene embeddings and return batch of images.
    """
    samples = []
    for i, row in data_df.iterrows():
        # multiply gene embedding by bitwise mask taken from sample mutations
        embedded_sample = row.values * gene_df.values
        samples.append(embedded_sample)
    return np.dstack(samples)

#---------------------------------------------------------------------------------------------------------
# Clustering Techniques
#---------------------------------------------------------------------------------------------------------
def mutation_spectral_clustering(mutation_df, num_clusters=25):
    """
    Determine a sorting on genes which creates visual structure.
    Calculates feature cooccurrence matrix, finds num_clusters as defined and sorts genes accordingly.
    """
    from sklearn.cluster import SpectralClustering
    c_matrix = onehot_df.T.dot(onehot_df) # cooccurrence matrix for genes in data source
    sc = SpectralClustering(num_clusters, affinity='precomputed', n_init=100, assign_labels='discretize')
    clusters = sc.fit_predict(c_matrix)
    return clusters


def genevec_spectral_clustering(genevec_df, num_clusters=25):
    """
    Determine a sorting on genes which creates visual structure.
    Calculates closest gene vectors, then finds num_clusters and sorts genes accordingly.
    """
    from sklearn.cluster import SpectralClustering
    
    return clusters

#---------------------------------------------------------------------------------------------------------
# Visualisation
#---------------------------------------------------------------------------------------------------------
def pca_2d(data):
    """
    2 component PCA helper
    """
    pca_2 = PCA(n_components=2)
    pca_2.fit(data)
    data_2=pca_2.transform(data)
    return data_2


def plot_pca_2d(data, title):
    """
    2 component PCA visualisation.
    """
    name="2 component PCA visualisation."
    pca_2 = PCA(n_components=2)
    pca_2.fit(data)
    data_2=pca_2.transform(data)
    
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('dimension 1', fontsize = 15)
    ax.set_ylabel('dimension 2', fontsize = 15)
    ax.set_title(title, fontsize = 20)
    ax.scatter(data_2[:,0], data_2[:,1], s = 5, alpha=0.5)
    ax.grid()
    
    if savefig:
        fig.savefig(FIGURES/'gene2vec'/name.lower().replace(' ','_'), dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)
    

def vis_pca2_genevec(savefig=True):
    """
    Run 2d PCA plot on all gene vectors to compare clustering and data distribution/information content.
    """
    
    fig = plt.figure(figsize=(18, 18))
    name="PCA2 plots on difference genevec model dimensions and iterations."; plt.suptitle(name)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    dims = [16, 128, 1024]
    iters = [1, 5, 10]
    
    plot_num = 1
    ax = plt.subplot(len(dims), len(iters), plot_num)
    for dim in dims:
        for iter_ in iters:
            # load data
            filename = str(GENE2VEC / 'dim_{}/iter_{}'.format(dim, iter_))
            wv, vocab = load_embeddings(filename)
            
            # run pca_2
            wv_2 = pca_2d(wv)

            # add subplot
            plt.subplot(len(dims), len(iters), plot_num, sharex=ax)
            plt.scatter(wv_2[:,0], wv_2[:,1], s = 5, alpha=0.1)
            plt.xlabel('dim1')
            plt.ylabel('dim2')
            plt.title(dim, size=18)
            
            plt.title('dim_{}/iter_{}'.format(dim, iter_), size=18)

            plot_num += 1

        if (dim == 64 and iter == 4):
            continue
    
    if savefig:
        fig.savefig(FIGURES/'gene2vec'/name.lower().replace(' ','_'), dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)
    

def vis_scree_genevec(savefig=True, ratio=True):
    """
    Run 2d PCA on gene vectors then compare scree decompositions.
    """
    fig = plt.figure(figsize=(18, 18))
    name = "Scree plots on difference genevec model dimensions and iterations."; plt.suptitle(name)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    dims = [16, 128, 1024]
    iters = [1, 5, 10]
    
    plot_num = 1
    for dim in dims:
        for iter_ in iters:
            # load data
            filename = str(GENE2VEC / 'dim_{}/iter_{}'.format(dim, iter_))
            wv, vocab = load_embeddings(filename)
            
            # run pca
            pca = PCA().fit(wv)
            
            # add subplot
            plt.subplot(len(dims), len(iters), plot_num)
            if ratio:
                plt.plot(np.cumsum(pca.explained_variance_ratio_))
            else:
                plt.plot(np.cumsum(pca.explained_variance_))
            plt.xlabel('number of components')
            plt.ylabel('cumulative explained variance');            
            plt.title('dim_{}/iter_{}'.format(dim, iter_), size=18)

            plot_num += 1

        if (dim == 64 and iter == 4):
            continue
    if savefig:
        fig.savefig(FIGURES/'gene2vec'/name.lower().replace(' ','_'), dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)

            
def visualise_sample(sample, savefig=True):
    """
    Visualise sample as image.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(pd.DataFrame(sample), aspect='auto')
    # plt.colorbar()
    

def visualise_clusters(data_df, samples=[3000, 5000, 7000], savefig=True):
    """
    Compare different clustering stategies.
    """
    # Initialise plot
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (18,12))
    name = 'TCGA Sample Number 5000'; fig.suptitle(name)
    
    clustered1 = apply_spectral_clustering(data_df[:,:,index:index+1], 
                                           onehot_df, num_clusters=5)
    
    clustered2 = apply_spectral_clustering(data_df[:,:,index:index+1], 
                                           onehot_df, num_clusters=10)
    
    clustered3 = apply_spectral_clustering(data_df[:,:,index:index+1], 
                                           onehot_df, num_clusters=20)
    
    clustered4 = apply_spectral_clustering(data_df[:,:,index:index+1], 
                                           onehot_df, num_clusters=50)


    axes[0,0].imshow(pd.DataFrame(np.absolute(clustered1[:,:,0])), aspect='auto', cmap='viridis')
    axes[0,0].set(title='num_clusters=5')
    axes[0,1].imshow(pd.DataFrame(np.absolute(clustered2[:,:,0])), aspect='auto', cmap='viridis')
    axes[0,1].set(title='num_clusters=10')
    axes[1,0].imshow(pd.DataFrame(np.absolute(clustered3[:,:,0])), aspect='auto', cmap='viridis')
    axes[1,0].set(title='num_clusters=20')
    axes[1,1].imshow(pd.DataFrame(np.absolute(clustered4[:,:,0])), aspect='auto', cmap='viridis')
    axes[1,1].set(title='num_clusters=50')
    
    if savefig:
        fig.savefig(FIGURES/'gene2vec'/name.lower().replace(' ','_'), dpi=fig.dpi, bbox_inches='tight', pad_inches=0.5)