from pathlib import Path

# Environment variables
HOME_LOCAL  = '/home/jovyan/ml_genomics'
HOME_REMOTE = '/home/athon/'
HOME        = HOME_REMOTE
THESIS      = HOME / Path('thesis/')

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
NROWS_TRAIN     = 1000
NROWS_VAL       = 1000
BATCH_SIZE      = 100

# Tokenisation - fixed
NGRAM_STRIDE    = [(3,1),(5,1),(7,1)]  #(ngram,stride) combinations for tokenisation

# Tokenisation - variable
MAX_VOCAB       = [4**3, 4**5, 4**7]
