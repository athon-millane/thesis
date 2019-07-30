from pathlib import Path

# Environment variables

THESIS      = Path('/home/athon/thesis')

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