import warnings
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.cluster.bicluster import SpectralBiclustering
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from itertools import cycle, islice
from joblib import Parallel, delayed

from gene2vec import *


def init_algos(X, n_clusters=10):
    """
    Initialise clustering algorithms including connectiivity matrix for X.
    """
    params = {'quantile': .3,
              'eps': .3,
              'damping': .9,
              'preference': -200,
              'n_neighbors': 10,
              'n_clusters': n_clusters,
              'min_samples': 20,
              'xi': 0.05,
              'min_cluster_size': 0.01}

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(X, n_neighbors=params['n_neighbors'], 
                                    include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)
        
    # Num clusters known
    two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
    ward = cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], 
                                           linkage='ward',
                                           connectivity=connectivity)
    spectral = cluster.SpectralClustering(n_clusters=params['n_clusters'], 
                                          eigen_solver='arpack',
                                          affinity="nearest_neighbors")
    average_linkage = cluster.AgglomerativeClustering(linkage="average", 
                                                      affinity="cityblock",
                                                      n_clusters=params['n_clusters'], 
                                                      connectivity=connectivity)
    birch = cluster.Birch(n_clusters=params['n_clusters'])
    gmm = mixture.GaussianMixture(n_components=params['n_clusters'], 
                                  covariance_type='full')
    
    # Num clusters unknown
    dbscan = cluster.DBSCAN(eps=params['eps'])
    optics = cluster.OPTICS(min_samples=params['min_samples'],
                            xi=params['xi'],
                            min_cluster_size=params['min_cluster_size'])
    

    clustering_algos = (
        ('MiniBatchKMeans', two_means),
        ('SpectralClustering', spectral),
        ('Ward', ward),
        ('DBSCAN', dbscan),
        ('OPTICS', optics),
        ('Birch', birch),
        ('GaussianMixture', gmm),
    )
    
    return clustering_algos
    

def cluster_job(args):
    """
    Cluster job for parallel compute to pass to plot.
    """
    name, algo, X_N = args
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        algo.fit(X_N)

    if hasattr(algo, 'labels_'):
        y_pred = algo.labels_.astype(np.int)
    else:
        y_pred = algo.predict(X_N)
        
    return y_pred


def vis_pca2_clusters(X, pca_degree, selected_algos, n_clusters=25, alpha=0.1):
    """
    Visualise PCA2 with cluster algorithms on different PCA degrees.
    """
    plt.figure(figsize=(24, 24))
    plt.suptitle("PCA2 Clusters with different clustering algorithms on different PCA degrees.")
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    pca_2 = PCA(n_components=2)
    pca_2.fit(X)
    X_2=pca_2.transform(X)
        
    plot_num = 1
    for n_components in pca_degree:
        pca_N = PCA(n_components=n_components)
        pca_N.fit(X)
        X_N=pca_N.transform(X)

        algos = init_algos(X_N, n_clusters=n_clusters)
        algos = tuple([algo for algo in algos if algo[0] in selected_algos])
        y_preds = Parallel()(delayed(cluster_job)((name, algo, X_N)) for name, algo in algos)
        
        for i, (name, algo) in enumerate(algos):
            y_pred = y_preds[i]
            colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                 '#f781bf', '#a65628', '#984ea3',
                                                 '#999999', '#e41a1c', '#dede00']),
                                          int(max(y_pred) + 1))))
            # add black color for outliers (if any)
            colors = np.append(colors, ["#000000"])

            plt.subplot(len(pca_degree), len(algos), plot_num)
            plt.title('{}, PCA dim = {}'.format(name, n_components))
            plt.scatter(X_2[:,0], X_2[:,1], s = 5, alpha=alpha, color=colors[y_pred])
            plot_num += 1
            
            
def vis_clustered_images(genes, dims, clustering_algos, n_clusters=25):
    """
    Visualise genes sorted by cluster with different clustering algos and genevec dimensions.
    """
    plt.figure(figsize=(24, 24))
    plt.suptitle("Images from different clustering algorithms and genevec model dimensions.")  
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    plot_num = 1
    for dim in dims:
        # load data
        filename = GENE2VEC_DIR + 'dim_{}/iter_10'.format(dim)
        wv, vocab = load_embeddings(filename)
        
        # get gene subset
        df_genevec = pd.DataFrame(wv.transpose(), columns=vocab)
        gene_subset = df_genevec[genes].values.transpose()
        
        # initialise algos on gene subset
        X = gene_subset
        algos = init_algos(X)
        algos = tuple([algo for algo in algos if algo[0] in clustering_algos])
        y_preds = Parallel()(delayed(cluster_job)((name, algo, X)) for name, algo in algos)
        
        other_algos = ['Unsorted', 'Spectral BiClustering']
        algos = clustering_algos + other_algos
        for i, name in enumerate(algos):
            if name == 'Unsorted':                      # Do nothing
                X = X
            elif (name == 'Spectral BiClustering'):     # Try out biclustering
                model = SpectralBiclustering(n_clusters=(4, 4), method='log',random_state=0).fit(X)
                X = X[np.argsort(model.row_labels_)]
                X = X[:, np.argsort(model.column_labels_)]
            else:                                       # Regular clustering gridsearch
                y_pred = y_preds[i]
                X = X[np.argsort(y_pred),:]
            
            plt.subplot(len(dims), len(algos), plot_num)
            plt.title('{}, GeneVec dim = {}'.format(name, dim))
            plt.imshow(pd.DataFrame(X.transpose()), aspect='auto')
            plot_num += 1
            

def vis_image_norm_colours(df_genevec, df_somatic, clusters, norms, colours, sample=5000):
    """
    Visualise how normalisation and colours alter appearance. 
    Apply to real data sample given by index.
    """
    plt.figure(figsize=(24, 24))
    plt.suptitle("Comparing colour maps and normalisation techniques.")
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    plot_num = 1
    for norm in norms:
        if (norm == 'none'):            # Leave X as is 
            image = df_somatic.transpose().values[:,sample] * df_genevec.values
            X = image[:,np.argsort(clusters)]
        elif (norm == 'non-zero'):      # 0,1 normalise all non zero data
            df_genevec_norm = MinMaxScaler().fit_transform(df_genevec.values)
            image = df_somatic.transpose().values[:,sample] * df_genevec_norm
            X = image[:,np.argsort(clusters)]
        elif (norm == 'abs value'):     # Take abs value of X
            image = df_somatic.transpose().values[:,sample] * df_genevec.values
            sorted_image = image[:,np.argsort(clusters)]
            X = np.absolute(sorted_image)
        
        for colour in colours:
            plt.subplot(len(norms), len(colours), plot_num)
            plt.title('Colour: {}, Norm: {}'.format(colour, norm))
            plt.imshow(pd.DataFrame(X), aspect='auto', cmap=colour)
            plot_num += 1            


def vis_norm_histogram(df_genevec, df_somatic, clusters, norms, sample=5000):
    """
    Visualise histogram of proposed normalisation techniques.
    """
    plt.figure(figsize=(16, 6))
    plt.suptitle("Distribution after proposed normalisation techniques.")
    
    plot_num = 1
    for norm in norms:
        if (norm == 'none'):            # Leave X as is 
            image = df_somatic.transpose().values[:,sample] * df_genevec.values
            X = image[:,np.argsort(clusters)]
        elif (norm == 'non-zero'):      # 0,1 normalise all non zero data
            df_genevec_norm = MinMaxScaler().fit_transform(df_genevec.values)
            image = df_somatic.transpose().values[:,sample] * df_genevec_norm
            X = image[:,np.argsort(clusters)]
        elif (norm == 'abs value'):     # Take abs value of X
            image = df_somatic.transpose().values[:,sample] * df_genevec.values
            sorted_image = image[:,np.argsort(clusters)]
            X = np.absolute(sorted_image)

        plt.subplot(1, len(norms), plot_num)
        plt.title('Norm: {}'.format(norm))
        plt.hist(X.flatten(), log=True, bins=20)
        plot_num += 1