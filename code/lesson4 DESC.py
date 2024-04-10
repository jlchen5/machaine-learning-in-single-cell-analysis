# DESC

## 0 Import python modules
import desc as desc
import numpy as np
import pandas as pd
import scanpy.api as sc
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline    # IPython的魔法函数，可以在IPython编译器里直接使用，作用是内嵌画图，省略掉plt.show()这一步，直接显示图像
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_versions()

## 1 Import data
### 1.1 Start from a 10X dataset
adata = desc.read_10X(data_path='ML_SC\\pbmc')       # load the 10X data by providing the path of the data

### 1.2 Start from *.mtx and *.tsv files
#### 1). Read the expression matrix from *.mtx file.
import pandas as pd
adata = desc.utilities.read_mtx('ML_SC\\pbmc\\matrix.mtx').T

#### 2). Read the *.tsv file for gene annotations. Make sure the gene names are unique.
genes = pd.read_csv('ML_SC\\pbmc\\genes.tsv', header=None, sep='\t')
adata.var['gene_ids'] = genes[0].values
adata.var['gene_symbols'] = genes[1].values
adata.var_names = adata.var['gene_symbols']
# Make sure the gene names are unique
adata.var_names_make_unique(join="-")

#### 3). Read the *.tsv file for cell annotations. Make sure the cell names are unique.
cells = pd.read_csv('ML_SC\\pbmc\\barcodes.tsv', header=None, sep='\t')
adata.obs['barcode'] = cells[0].values
adata.obs_names = cells[0]
# Make sure the cell names are unique
adata.obs_names_make_unique(join="-")

### 1.3 Start from a *.h5ad file
adata = desc.read_h5ad('data/pbmc.h5ad')
#or use 
#adata=sc.read_h5ad("data/pbmc.h5ad")


## 2 Preprocessing
### 2.1 Filtering cells and genes
adata
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

mito_genes = adata.var_names.str.startswith('MT-')
# for each cell compute fraction of counts in mito genes vs. all genes
# the `.A1` is only necessary as X is sparse (to transform to a dense array after summing)
adata.obs['percent_mito'] = np.sum(
    adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
# add the total counts per cell as observations-annotation to adata
adata.obs['n_counts'] = adata.X.sum(axis=1).A1

sc.pl.violin(adata, ['n_genes', 'n_counts', 'percent_mito'],jitter=0.4, multi_panel=True)

adata = adata[adata.obs['n_genes'] < 2500, :]
adata = adata[adata.obs['percent_mito'] < 0.05, :]

### 2.2 Normalization
desc.normalize_per_cell(adata, counts_per_cell_after=1e4)
#or use
#sc.pp.normalize_per_cell(adata,counts_per_cell_after=1e4)

### 2.3 Logarithm transformation
desc.log1p(adata)
#or use
#sc.pp.log1p(adata)
adata.raw=adata

### 2.4 Selection of highly variable genes
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, subset=True)
adata = adata[:, adata.var['highly_variable']]
adata

### 2.5 Scaling
desc.scale(adata, zero_center=True, max_value=3)
#or use
#sc.pp.scale(adata, zero_center=True, max_value=3)

## 3 Desc analysis
adata = desc.train(adata, dims=[adata.shape[1], 32, 16], tol=0.005, n_neighbors=10,
                   batch_size=256, louvain_resolution=[0.8],
                   save_dir="result_pbmc3k", do_tsne=True, learning_rate=300,
                   do_umap=True, num_Cores_tsne=4,
                   save_encoder_weights=True)

## 4 Visualization
prob_08=adata.uns["prob_matrix0.8"]
adata.obs["max.prob0.8"]=np.max(prob_08,axis=1)
#tSNE plot 
sc.pl.scatter(adata,basis="tsne0.8",color=['desc_0.8',"max.prob0.8",'MS4A1', 'NKG7'])
#Umap plot 
sc.pl.scatter(adata,basis="umap0.8",color=['desc_0.8',"max.prob0.8",'MS4A1', 'NKG7'])
adata.obs["desc_0.8"]=adata.obs["desc_0.8"].astype(str)
sc.pl.stacked_violin(adata, ["MS4A1", "GNLY", "CD3E", "CD14", "FCER1A", "FCGR3A", "LYZ", "PPBP", "CD8A"],
                     groupby='desc_0.8',figsize=(8,10),swap_axes=True)

## 5 Save the result 
### 5.1 Save to a *.h5ad file
adata.write('../result/desc_result.h5ad')

### 5.2 Save to *.csv files
#`obs` slot
meta_data=adata.obs.copy()
meta_data.to_csv("meta.data.csv",sep=",")
#`obsm` slot, which is numpy.ndarray
obsm_data=pd.DataFrame(adata.obsm["X_tsne0.6"])
obsm_data.to_csv("tsne.csv",sep=",")













