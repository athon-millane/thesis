# Environment variables for running pretraining on xlnet
import os
from pathlib import Path

#------- paths -------#
HOME         = Path('/home/jovyan/ml_genomics')
IMDB         = HOME / ('/xlnet/data/imdb/aclImdb')
IMDB_DATA    = IMDB / ('train/unsup/*.txt')

GENOME       = HOME / ('genomeXL/data/human')
# version of reference genome
GRCH38_P13   = GENOME / ('GCF_000001405.39_GRCh38.p13_genomic.fna')

#--- env variables ---#
# preprocessing text data with trained sentencepiece model (English text)


# sentencepiece applied to genomic data
os.environ["MODEL_PREFIX"]  = "sp10m.cased.v3"
os.environ["INPUT"]         = "genomeXL/data/genome.txt"

# pretraining on text data
os.environ["INFO_DIR"]      = str(IMDB / 'tfrecords')