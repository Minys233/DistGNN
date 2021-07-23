import shutil
from functools import partial
from pathlib import Path

import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import nn

from lib.dataset import StructureSearchDataModuleChannel, seed_everything
from lib.dataset import savedir
from lib.model import GATNet, GATENet, LSTMNet
from lib.model_util import GlobalAddPool, NodeAttentionPool, StructureAttentionPool

seed_everything(42)

# Dataset
pdbdir = None
tmfile = None
thres = -1


# positive condition, sample as positive and anchor
def pos(x): return x > 0.6


val_frac = 0.1
batch_size = 32  # max value fit GPU, modify this for your own GPU
fold = 100  # specify in cmd line, if not it will raise error

# Model
# these are shared parameters
enum_dict = {8: 13, 10: 17, 12: 21, 15: 27}
enum = -1
hid = 256
fout = 256
convlayer = 5
drop = 0.2
actfn = partial(F.leaky_relu, negative_slope=0.1, inplace=True)
gat_heads = 3
edge_emb = 8
jk_layer = 1

margin = 3
norm = 2.0
k = 2
sample_from = 0.20
max = 282

# GATENet has no other kwargs
gate_kw = dict()
# LSTMNet
lstm_kw = {'lstm_layer': 2}

datamodule = None
dset = None


def setup_dataset():
    global datamodule, dset
    datamodule = StructureSearchDataModuleChannel(pdbdir, tmfile, thres, pos, val_frac, batch_size, fold)
    dset = [datamodule.train_dataset, datamodule.valid_dataset, datamodule.test_dataset, datamodule.full_dataset]


config_kwargs = {
    'thres': thres,
    'val_frac': val_frac,
    'batch_size': batch_size,
    'fold': fold,

    'enum': enum, 'hid': hid, 'fout': fout,
    'convlayer': convlayer, 'drop': drop,
    'actfn': actfn, 'gat_heads': gat_heads,
    'edge_emb': edge_emb, 'jk_layer': jk_layer,
    'margin': margin, 'norm': norm,
    'k': k, 'sample_from': sample_from,
    'max': max
}


# setup model
def dict2str(d: dict):
    s = ""
    for k, v in d.items():
        s += f"_{k}-{v}"
    return s


def setup_gnn_logger(model, pool, log=True):
    global enum
    enum = enum_dict[int(thres)]
    if isinstance(actfn, partial):
        param = f"_{enum}_{hid}_{fout}_{convlayer}_{drop}_{actfn.func.__name__}_{gat_heads}_{edge_emb}_{jk_layer}"
    else:
        param = f"_{enum}_{hid}_{fout}_{convlayer}_{drop}_{actfn.__name__}_{gat_heads}_{edge_emb}_{jk_layer}"

    if model == 'GATNet':
        net = GATNet(enum, hid, fout, convlayer, drop, actfn, gat_heads, edge_emb, jk_layer)
        other = ""
    elif model == 'GATENet':
        net = GATENet(enum, hid, fout, convlayer, drop, actfn, gat_heads, edge_emb, jk_layer)
        other = ""
    elif model == 'LSTMNet':
        net = LSTMNet(enum, hid, fout, convlayer, drop, actfn, gat_heads, edge_emb, jk_layer, **lstm_kw)
        other = dict2str(lstm_kw)
    else:
        raise NotImplementedError(f"We do not have this gnn: {model}")

    if pool == 'GlobalAddPool':
        net_pool = GlobalAddPool()
    elif pool == 'NodeAttentionPool':
        net_pool = NodeAttentionPool(nn.Linear(fout, 1), nn.Linear(fout, fout))
    elif pool == 'StructureAttentionPool':
        net_pool = StructureAttentionPool(fout)
    else:
        # raise NotImplementedError(f"We do not have this pool: {pool}")
        net_pool = None
    if log:
        desc = f"{model}_{pool}" + param + other + f'_FOLD-{fold}'
        save = savedir(desc)
        print("Logging into:", save)
        if Path(save).is_dir():
            # shutil.rmtree(save)
            print("deleting", save, "first, since it exists!")
        else:
            print("Moving previous ckpt dir")
            ckptdir = Path(
                '.') / 'result' / f'train-extra-{int(thres)}-continue' / f'retrieval-lstmnet-{int(fold)}-{int(thres)}'
            shutil.copytree(ckptdir, save, dirs_exist_ok=True)
        logger = SummaryWriter(logdir=save)
    else:
        logger = None
    config_kwargs['model'] = model
    config_kwargs['pool'] = pool
    return net, net_pool, logger

