from pathlib import Path
import multiprocessing

# Environment variables
HOME_LOCAL  = '/home/jovyan/ml_genomics'
HOME_REMOTE = '/home/athon/'
HOME        = HOME_REMOTE
THESIS      = HOME / Path('thesis/')

N_CPUS      = multiprocessing.cpu_count()

#----------------------------------------------------
# part 1
# data
TCGA        = THESIS / ('data/tcga')
TCGA_MUT    = TCGA / 'processed_somatic_mutations_subset.csv'

DRIVERS     = THESIS / ('data/drivers')
INTOGEN     = DRIVERS / 'intogen-drivers-data.tsv'
DRIVERS_ALL = DRIVERS / 'drivers.xlsx'
DRIVER_LIST = DRIVERS / 'drivers.csv'

# models
MODELS      = THESIS / ('models')
GENE2VEC    = MODELS / 'gene2vec'
SKLEARN     = MODELS / 'sklearn-ensemble'
FASTAI      = MODELS / 'fastai-cnn'

#------------------------------------------------------
# part 2
HUMAN       = THESIS / Path('data/human/')

# Dataset params
NROWS_TRAIN     = 20000
NROWS_VAL       = 20000

# Tokenisation - fixed
NGRAM_STRIDE    = [(3,3),(5,5),(7,7)]  #(ngram,stride) combinations for tokenisation

# Tokenisation - variable
MAX_VOCAB       = [4**3, 4**5, 4**7]

# Batch sizes
BS              = [3200, 1600, 600]
