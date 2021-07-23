import argparse

import pandas as pd
import torch

import configure as conf

parser = argparse.ArgumentParser()
parser.add_argument('model', choices=['GATENet', 'LSTMNet'])
parser.add_argument('pool', choices=['GlobalAddPool', 'NodeAttentionPool', 'StructureAttentionPool'])
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--ckpt', type=str, default="")
parser.add_argument('--fold', type=int, default=-1)
args = parser.parse_args()


class Inferencer(torch.nn.Module):
    def __init__(self, gnn, pool):
        super(Inferencer, self).__init__()
        self.gnn = gnn
        self.pool = pool

    def forward(self, batch):
        nembed = self.gnn(batch)
        return nembed


conf.fold = args.fold if args.fold >= 0 else None
conf.setup_dataset()
gnn, pool, _ = conf.setup_gnn_logger(args.model, args.pool, log=False)
model = Inferencer(gnn, pool).to(args.device)
if args.ckpt:
    ckpt = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(ckpt['state_dict'])
metrics = conf.dset[2].evaluate_val(model, args.device, 'euclidean', 'plain').T

eval_list = conf.dset[2].names
pos_pair = conf.dset[2].pos_pair
pdblst = [i for i in eval_list if i in pos_pair]
print(len(pdblst), metrics.shape)

k = [1, 5, 10, 20, 50, 100]
roc, prc = metrics[0], metrics[1]
hitk = metrics[2:8]
d1 = {f'hit@top{n}': h for n, h in zip(k, hitk)}
prec = metrics[8:14]
d2 = {f'prec@top{n}': p for n, p in zip(k, prec)}
recall = metrics[14:20]
d3 = {f'recall@top{n}': r for n, r in zip(k, recall)}

print(len(pdblst), len(roc), len(prc))

df = pd.DataFrame({'name': pdblst, 'auroc': roc, 'auprc': prc, **d1, **d2, **d3})
df.to_csv(f'resultDistGNN-{conf.fold}.csv', index=False)
