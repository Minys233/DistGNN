import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn.functional import mse_loss

import configure as conf


def alignment_loss(aln_dict, qbatch, gbatch, nembed, a, p):
    loss = 0.0
    pos = set([(aa, pp) for aa, pp in zip(a, p)])
    tmpdict = dict()
    for aa, pp in pos:
        if not aln_dict[(qbatch[aa], qbatch[pp])]:
            continue
        idx1, idx2 = aln_dict[(qbatch[aa], qbatch[pp])]
        s1 = gbatch.ptr[aa].item()
        s2 = gbatch.ptr[pp].item()
        tmpdict[(aa, pp)] = (s1 + idx1, s2 + idx2)

    for aa, pp in zip(a, p):
        if not (aa, pp) in tmpdict:
            continue
        index1, index2 = tmpdict[(aa, pp)]
        loss += mse_loss(nembed[index1], nembed[index2])
    return loss


def print_logging(epoch, trainmetrics, validmetrics, testmetrics, pbar=None, header=False, valid=False, test=False):
    pfn = pbar.write if pbar is not None else print
    if header:
        pfn("_" * 197)
        pfn("| Epoch | Type  | AUROC  | AUPRC  | Hit@1  | Hit@5  | Hit@10 | Hit@20 | Hit@50 | Hit@100| Pre@1  | Pre@5  "
            "| Pre@10 | Pre@20 | Pre@50 | Pre@100| Rec@1  | Rec@5  | Rec@10 | Rec@20 | Rec@50 | Rec@100|")
        pfn("-" * 197)

    pfn(f"| {epoch:<5d} | Train | " + " | ".join([f'{v:.4f}' for v in trainmetrics]) + ' |')
    if valid:
        pfn(f"| {epoch:<5d} | Vali  | " + " | ".join([f'{v:.4f}' for v in validmetrics]) + ' |')
    if test:
        pfn(f"| {epoch:<5d} | Test  | " + " | ".join([f'{v:.4f}' for v in testmetrics]) + ' |')
    pfn("-" * 197)


class Trainer(nn.Module):
    def __init__(self, margin, gnn, pool, lossfn, dset=None, logger=None, device='cpu'):
        super().__init__()
        self.margin = margin
        self.gnn = gnn
        self.pool = pool
        self.lossfn = lossfn
        # evaluate
        self.dset = dset
        self.logger = logger
        self.device = device
        self.global_step = 0

        if dset is None:
            raise ValueError("pass datasets here in order to evaluation! (trainset, validation set)")
        if logger is None:
            raise ValueError("pass tensorboardX logger here in order to evaluation!")

    def forward(self, batch):
        return self.gnn(batch)

    def training_step(self, batch, **kwargs):
        (a, p, n), qbatch = self.dset[0].sample_batch_triplet(self, batch, self.margin, **kwargs)
        if len(a) > 0:
            gbatch = self.dset[0].get_graphs(qbatch).to(self.device)
            nembed = self(gbatch)
            gembed = self.pool(nembed, gbatch.batch, gbatch.ptr)
        else:
            return None, 0, 0
        loss = self.lossfn(gembed[a], gembed[p],
                           gembed[n])  # + alignment_loss(self.dset[0].aligndict, qbatch, gbatch, nembed, a, p)
        logger.add_scalar('train/train-loss', loss.item(), global_step=self.global_step)
        logger.add_scalar('train/Nencode', len(qbatch), global_step=self.global_step)
        logger.add_scalar('train/Ntriplet', len(a), global_step=self.global_step)
        self.global_step += 1
        return loss, len(qbatch), len(a)

    def shared_evaluate(self, metrics, post):
        k = [1, 5, 10, 20, 50, 100]
        metrics = metrics.mean(axis=0)
        # roc, prc, hit@k(6), prec@k(6), recall@k(6)
        roc, prc = metrics[0], metrics[1]
        self.logger.add_scalar(post + '/auroc-' + post, roc, global_step=self.global_step)
        self.logger.add_scalar(post + '/auprc-' + post, prc, global_step=self.global_step)

        hitk = metrics[2:8]
        self.logger.add_scalars(post + '/hit@topk-' + post, {f'hit@top{n}': h for n, h in zip(k, hitk)},
                                global_step=self.global_step)
        prec = metrics[8:14]
        self.logger.add_scalars(post + '/prec@topk-' + post, {f'prec@top{n}': p for n, p in zip(k, prec)},
                                global_step=self.global_step)
        recall = metrics[14:20]
        self.logger.add_scalars(post + '/recall@topk-' + post, {f'recall@top{n}': r for n, r in zip(k, recall)},
                                global_step=self.global_step)
        return metrics

    def train_evaluate(self, metric='euclidean'):
        metrics = self.dset[0].evaluate_train(self, self.device, metric)
        return self.shared_evaluate(metrics, post='train')

    def valid_evaluate(self, metric='euclidean'):
        metrics = self.dset[1].evaluate_val(self, self.device, metric)
        return self.shared_evaluate(metrics, post='valid')

    def test_evaluate(self, metric='euclidean'):
        metrics = self.dset[2].evaluate_val(self, self.device, metric)
        return self.shared_evaluate(metrics, post='test')

    def configure_optimizers(self, lr):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', 0.5, 2, verbose=True, threshold_mode='abs')
        return optimizer, scheduler


def step(auprc, schedule):
    schedule.step(auprc)


def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['GATNet', 'GATENet', 'LSTMNet'])
    parser.add_argument('pool', choices=['GlobalAddPool', 'NodeAttentionPool', 'StructureAttentionPool'])
    parser.add_argument('sample', choices=['hardpos', 'hardneg', 'hardest', 'easypos_hardneg'])
    parser.add_argument('thres', type=int, choices=[8, 10, 12, 15])
    parser.add_argument('device')
    parser.add_argument('--ckpt', type=str, default="")
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--fold', type=int, default=None)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--batchsize', default=32, type=int)

    args = parser.parse_args()
    args_dict = vars(args)
    print(args_dict)

    device = args.device
    conf.thres = args.thres
    conf.fold = args.fold
    conf.batch_size = args.batchsize
    conf.setup_dataset()

    gnn, pool, logger = conf.setup_gnn_logger(args.model, args.pool)
    lossfn = nn.TripletMarginLoss(conf.margin, p=conf.norm)
    model = Trainer(conf.margin, gnn, pool, lossfn, conf.dset, logger, device).to(device)
    opt, lrschedule = model.configure_optimizers(args.lr)

    if args.ckpt:
        print("Resuming ckpt from:", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=args.device)
        model.load_state_dict(ckpt['state_dict'])
    else:
        print("Train from scratch")

    maxepoch = 0
    for ckpt in Path(logger.logdir).glob('**/*.pth'):
        if ckpt.name.startswith('model'):
            try:
                e = int(ckpt.name.split('.')[0].split('-')[1])
            except:
                continue
            if e > maxepoch:
                maxepoch = e
    print("Loading", Path(logger.logdir) / f'model-{maxepoch}.pth')
    ckpt = torch.load(Path(logger.logdir) / f'model-{maxepoch}.pth', map_location=args.device)
    model.load_state_dict(ckpt['state_dict'])
    args.epoch = maxepoch + 1
    print("Starting from", args.epoch, "epoch")

    train_metrics = model.train_evaluate()
    valid_metrics = model.valid_evaluate()
    test_metrics = model.test_evaluate()
    print_logging(0, train_metrics, valid_metrics, test_metrics, pbar=None, header=True, valid=True, test=True)
    torch.save({'state_dict': model.state_dict(),
                'train-prc': train_metrics[1], 'valid-prc': valid_metrics[1], 'test-prc': test_metrics[1],
                **args_dict, **conf.config_kwargs},
               Path(logger.logdir) / f"model-untrained.pth")

    for epoch in range(args.epoch, 103):
        trainloader = conf.datamodule.train_dataloader()  # train set size 13067
        print(f"### Epoch {epoch} ###")
        encode_hist = []
        for idx, nbatch in enumerate(trainloader):
            opt.zero_grad()
            loss, nencode, ntriplet = model.training_step(nbatch, k=conf.k, sample_from=conf.sample_from,
                                                          device=device, **{args.sample: True}, max=conf.max)
            if loss is not None:
                encode_hist.append(int(nencode))
                if idx % 20 == 0:
                    print(
                        f"{idx}/{len(trainloader.dataset) // trainloader.batch_size + 1} - loss: {loss.item()}, Nenc: {int(nencode)}, Ntri: {int(ntriplet)}, Nimp: {int(lrschedule.num_bad_epochs)}")
                loss.backward()
                opt.step()

        if not epoch % 3:
            train_metrics = model.train_evaluate()
            valid_metrics = model.valid_evaluate()
            test_metrics = model.test_evaluate()
            print_logging(epoch, train_metrics, valid_metrics, test_metrics, pbar=None, valid=True, test=True)
            torch.save({'state_dict': model.state_dict(),
                        'train-prc': train_metrics[1], 'valid-prc': valid_metrics[1], 'test-prc': test_metrics[1],
                        **args_dict, **conf.config_kwargs},
                       Path(logger.logdir) / f"model-{epoch}.pth")
            step(train_metrics[1], lrschedule)

        if conf.max is None:
            conf.max = int(np.max(encode_hist)) + 1

        if np.mean(encode_hist) < conf.max // 2:
            coef = 2 ** 0.5
            conf.batch_size = conf.batch_size * coef
            conf.datamodule.batch_size = int(conf.batch_size)
            trainloader = conf.datamodule.train_dataloader()
