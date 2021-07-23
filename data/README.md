# Data for training and testing DistGNN

## File description
1. `train-i.txt` and `test-i.txt`

Family-level stratified split protein structures, 
used in 10-fold cross validation. These files are
generated with random seed 42. Family labels are stored in 
`full-id-cla.txt`, which is from official site of SCOP 2.06.

Specially, `test-2.07.txt` is the test set for newly released
proteins, curated from the subtraction of SCOP 2.07 and SCOP 2.06.

2. `*.pth` files

These files are directly used for train DistGNN, they are preprocessed
from SCOP-2.06. They are too large to store in github repo,
you can download them from [Tsinghua Cloud Drive](https://cloud.tsinghua.edu.cn/d/a0af9663b7a8429e86b4/)
The contents are defined as follows:

- `StructureSearchDataModuleChannel_graphdict_i.pth`: Protein graphs with distance cutoff `i`.
  In the paper, we use `i=10`. Other files are used for parameter searching.
  
- `StructureSearchDataModuleChannel_pair.pth`: Positive pairs are recorded in the file.

- `StructureSearchDataModuleChannel_pair.pth`: Aligned residue indices of positive pairs.


