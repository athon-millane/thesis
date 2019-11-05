import pandas as pd, matplotlib.pyplot as plt
from .config import INTOGEN,DRIVER_LIST

def drop_low_mut_count(df, feature, cutoff=100):
    """
    Drop rows which contain features which occur less than cutoff times in the dataset.
    """
    subsample = df[feature].value_counts()[(df[feature].value_counts() > cutoff)].index.tolist()
    return df[df[feature].isin(subsample)]


def merge_label(df, label1, label2, merged_label):
    """
    Merge label1 and label2 into merged label within dataframe.
    """
    df.loc[(df['project_short_name'] == label1) | 
           (df['project_short_name'] == label2), 'project_short_name'] = merged_label
    return df


def process_labels(df):
    """
    Merge cancers that are established clinically to be the same.
    """
    # Colon and Rectal cancers are now considered the same cancer
    # COAD, READ -> COADREAD
    df = merge_label(df, 'TCGA-COAD', 'TCGA-READ', 'MERGE-COADREAD')
    
    # GBM and LGG are both forms of brain Glioma
    # GBM, LGG   -> GBMLGG
    df = merge_label(df, 'TCGA-GBM', 'TCGA-LGG', 'MERGE-GBMLGG')
    
    # Stomach and Esophegal cancers are also considered the same
    # ESCA, STAD -> STES
    df = merge_label(df, 'TCGA-ESCA', 'TCGA-STAD', 'MERGE-STES')
    
    return df


def generate_drivers_list():
    """
    Generate condensed drivers list from all driver Excel spreadsheet.
    """
    import pandas as pd

    sheets = {
        'mutsigcv':'MutsigCV',
        '2020+'   :'2020+'
    }

    drivers_df = pd.DataFrame()
    for driver,sheet in sheets.items():
        df = pd.read_excel(DRIVERS_ALL, sheetname=sheet, header=1)
        drivers_df[driver + '_genes'] = df['gene']

    drivers_df.to_csv(DRIVER_LIST, index=False)


def filter_genes(df, by='intogen', number=1348):
    """
    Filter only genes that intersect with listed drivers from Intogen.
    """
    if by == 'intogen':
        intogen_drivers = pd.read_csv(INTOGEN, sep='\t')
        driver_genes = intogen_drivers['SYMBOL'].head(number).tolist()
    
    elif by == 'mutsigcv' or by == '2020+':
        driver_genes = pd.read_csv(DRIVER_LIST)[by + '_genes'].head(number).tolist()
    
    return df[df['Hugo_Symbol'].isin(driver_genes)]


def filter_variants(df):
    """
    Filter out variants according to a list provided by Dr Nic Waddel (QIMR).
    """
    
    waddell_list = ['missense_variant',
                    'stop_gained',
                    'frameshift_variant',
                    'splice_acceptor_variant',
                    'splice_donor_variant',
                    'start_lost',
                    'inframe_deletion',
                    'inframe_insertion',
                    'stop_lost']
    
    return df[df['One_Consequence'].isin(waddell_list)]


def dedup(df_in):
    """
    Deduplicate gene sample combinations with >1 mutations and aggregate 
    with additional feature of variant count for gene sample combination.
    """
    df = df_in.copy()
    
    counts = df.groupby('case_barcode')['Hugo_Symbol'].value_counts()
    df = df.drop_duplicates(subset=['case_barcode', 'Hugo_Symbol'])
    df = df.set_index(['case_barcode', 'Hugo_Symbol'])
    df['mutation_count'] = counts
    df = df.reset_index()
    
    return df


def convert_to_onehot(df_in):
    """
    Convert count encoding to one-hot encoded representation of df.
    """
    df = df_in.copy()
    df[df != 0] = 1
    return df


def reshape_pivot(df_in):
    """
    Reduce df to crucial subset then pivot on cases and genes.
    """
    df = (df_in[['case_barcode', 'Hugo_Symbol', 'mutation_count']]
              .copy()
              .pivot(index='case_barcode', columns='Hugo_Symbol', values='mutation_count')
              .fillna(0)
              .astype(int))
    
    return df


def get_label_df(df_in, df_X):
    """
    Get label df from flat processed df.
    """
    df_y = (df_in.loc[df_in['case_barcode'].isin(df_X.index)]
                 .groupby('case_barcode')
                 .head(1)
                 .set_index('case_barcode')[['project_short_name']]
                 .sort_index())
    return df_y


def visualise_distributions(df, title):
    """
    Plot distribution and frequency of features of interest for raw and processed TCGA df.
    """
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (16,8))
    fig.suptitle(title)
    plt.subplots_adjust(hspace=0.6)
    
    df.groupby('case_barcode').head(1)['project_short_name'].value_counts() \
        .plot(kind='bar', title='Cases per Cancer type', ax=axes[0,0], color='m')
    
    df['Variant_Classification'] \
        .value_counts().plot(kind='bar', title='Variants per variant type', ax=axes[1,0], logy=True, color='g')
    
    df['case_barcode'].value_counts() \
        .plot(title='Log Variants per case, {0:d} cases'
              .format(df['case_barcode'].value_counts().shape[0]), 
              ax=axes[0, 1],  logy=True, color='r')
    
    df['Hugo_Symbol'].value_counts() \
        .plot(title='Log Variants per gene, {0:d} genes'
              .format(df['Hugo_Symbol'].value_counts().shape[0]), 
              ax=axes[1, 1], logy=True, color='b')