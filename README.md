# DistGNN
This is the github repo that implements the DistGNN.


## Requirements and 3rd-party software
Refer to `lib/dataset.py`, we use DSSP and MSMS. please download
correct version and put them in `third_party` folder.

## Train from scratch
We do not recommand to train DistGNN from scratch, but if you
would like to do it, first download all data files, referring to
the readme file in `data`. Then download all requirements refer to scipts.
please use python 3.7 or 3.8 and the newest version of libraries.
We will soon release the environment files to setup environment easily.

Then run the following command:
```bash
python train_retrieval.py --fold x --batchsize y LSTMNet StructureAttentionPool hardneg 10 cuda:z
```

This will log all files in your `~/runs` folder. You could use a small
batchsize to train the DistGNN, the performances should be no close to ones we provided.


## Results of DistGNN and baseline models
For simplicity, we provide only the baseline results on SCOP40-2.06
and SCOP40-2.07, in `baselines` folder. It is important to note that, for ContactLib,
[the original code](https://github.com/louchenyao/contactlib) works only on proteins in .pdb
extension, and contains no punctuations in name. This will make the performances drops greatly.
So we modify the code and make it works fine on SCOP structures.

