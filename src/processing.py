def process_fasta(fname, c1, c2, filter_txt=None):
    from Bio import SeqIO
    genome = SeqIO.parse(fname, 'fasta')
    if filter_txt:
        chroms = [GB for GB in genome if 'NC_' in GB.id]
    else:
        chroms = [GB for GB in genome]
    genome = ''.join([i.seq.__str__() for i in chroms]).upper()
    genome_chunks = [genome[i:i+c1] for i in range(0, len(genome), c1) if not 'N' in genome[i:i+c1] and set(genome[i:i+c1])==set('ATGC')]
    clean_genome = ''.join(genome_chunks)
    data = [clean_genome[i:i+c2] for i in range(0, len(clean_genome), c2)]
    
    return data

def process_txt_sentencepiece():
    return