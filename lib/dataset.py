import itertools
import random
import warnings
from pathlib import Path

import dill
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from Bio.Data.IUPACData import protein_letters_1to3
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.HSExposure import HSExposureCA
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.ResidueDepth import ResidueDepth
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch_geometric.data import Data, Dataset, Batch, DataLoader
from tqdm import tqdm

DATA = Path(__file__).absolute().parent.parent / 'data'
DSSPBIN = Path(__file__).absolute().parent.parent / 'third_party' / 'dssp-3.1.4' / 'dssp'
MSMSBIN = Path(__file__).absolute().parent.parent / 'third_party' / 'msms-2.6.1' / 'msms.x86_64Linux2.2.6.1'


def seed_everything(seed: int = None):
    import os
    import random
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min
    try:
        if seed is None:
            seed = os.environ.get("SEED")
        seed = int(seed)
        print(f"Setting by env var $SEED={seed}")
    except (TypeError, ValueError):
        seed = random.randint(min_seed_value, max_seed_value)
        print(f"No correct seed found, seed set to {seed}")

    if not (min_seed_value <= seed <= max_seed_value):
        print(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
        seed = random.randint(min_seed_value, max_seed_value)

    print(f"Global seed set to {seed}")
    os.environ["SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def metrics_at_k(gt, score, topk):
    sidx = score.argsort()[::-1]  # high to low
    gt, score = gt[sidx], score[sidx]
    numtrue = gt.sum()
    hitk = [0 for _ in topk]
    prec = [0 for _ in topk]
    recl = [0 for _ in topk]
    for i, k in enumerate(topk):
        numtrue_k = gt[:k].sum()
        hitk[i] = 1 if numtrue_k > 0 else 0
        prec[i] = numtrue_k / k
        recl[i] = numtrue_k / numtrue
    return hitk, prec, recl


def savedir(comment=''):
    # current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    logdir = Path.home() / 'runs' / comment
    return logdir


class ProteinData:
    DSSP_SS8 = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-']
    DSSP_SS8toSS3 = {'H': 'H', 'G': 'H', 'I': 'H', 'E': 'E', 'B': 'E', 'S': 'C', 'T': 'C', '-': 'C'}
    DSSP_SS3 = ['H', 'E', 'C']
    # 310 helix (G), α-helix (H), π-helix (I), β-stand (E), bridge (B), turn (T), bend (S), and others (C).
    AA_3LETTER = [aa.upper() for aa in protein_letters_1to3.values()] + ['UNK', 'XXX']

    @staticmethod
    def onehot_encode(aa, ss):
        idxaa = torch.zeros(len(aa), dtype=torch.long)
        idxss = torch.zeros(len(ss), dtype=torch.long)
        for ii, (a, s) in enumerate(zip(aa, ss)):
            try:
                idxaa[ii] = ProteinData.AA_3LETTER.index(a)
            except ValueError:
                idxaa[ii] = ProteinData.AA_3LETTER.index('XXX')
            idxss[ii] = ProteinData.DSSP_SS3.index(s)
        feataa = F.one_hot(idxaa, len(ProteinData.AA_3LETTER))
        featss = F.one_hot(idxss, len(ProteinData.DSSP_SS3))
        return torch.cat((feataa, featss), dim=1)

    @staticmethod
    def parse_pdb(pdbpath):
        pdbpath = str(pdbpath)
        res_lst, res_syb, coords, ss = [], [], [], []
        asa, rasa, rd, hseau, hsead = [], [], [], [], []
        # Secondary Structure, DSSP, sudo apt-get install dssp
        warnings.simplefilter('ignore', PDBConstructionWarning)
        parser = PDBParser()
        structure = parser.get_structure('id', pdbpath)
        # use the first model
        model = structure[0]
        try:
            dssp = DSSP(model, pdbpath, dssp=str(DSSPBIN))
            # some proteins have only CA coords, which fails DSSP, don't consider them now
        except Exception as e:
            print(e.args, pdbpath)
            raise NotImplementedError(f"{pdbpath}, please add to `FAIL`, it failed to run DSSP")

        try:
            ResidueDepth(model, msms_exec=str(MSMSBIN))
        except Exception as e:
            print(e.args, pdbpath)
            print(f"{pdbpath}, please add to `FAIL`, it failed to run MSMS")

        try:
            HSExposureCA(model)
        except Exception as e:
            print(e.args, pdbpath)
            print(f"{pdbpath}, please add to `FAIL`, it failed to run HSE_CA")

        for chain in model.child_list:
            for res in chain:
                if 'CA' in res:  # normal residue with CA atom
                    coords.append(res['CA'].coord)
                    res_lst.append(res)
                    res_syb.append(res.get_resname())
                    if 'EXP_DSSP_ASA' in res.xtra and isinstance(res.xtra['EXP_DSSP_ASA'], (int, float)):
                        asa.append(res.xtra['EXP_DSSP_ASA'])
                    else:
                        asa.append(-1.0)
                    #                         print(f"-1 ASA @ {res} of {pdbpath}")

                    if 'EXP_DSSP_RASA' in res.xtra and isinstance(res.xtra['EXP_DSSP_RASA'], (int, float)):
                        rasa.append(res.xtra['EXP_DSSP_RASA'])
                    else:
                        rasa.append(-1.0)
                    #                         print(f"-1 RASA @ {res} of {pdbpath}")

                    if 'EXP_RD' in res.xtra:
                        rd.append(res.xtra['EXP_RD'])
                    else:
                        rd.append(-1.0)
                    #                         print(f"-1 RD @ {res} of {pdbpath}")

                    if 'EXP_HSE_A_U' in res.xtra:
                        hseau.append(res.xtra['EXP_HSE_A_U'])
                    else:
                        hseau.append(-1.0)
                    #                         print(f"-1 HSE_U @ {res} of {pdbpath}")

                    if 'EXP_HSE_A_D' in res.xtra:
                        hsead.append(res.xtra['EXP_HSE_A_D'])
                    else:
                        hsead.append(-1.0)
                    #                         print(f"-1 HSE_D @ {res} of {pdbpath}")
                    if 'SS_DSSP' in res.xtra:
                        ss.append(ProteinData.DSSP_SS8toSS3[res.xtra['SS_DSSP']])
                    else:
                        ss.append(ProteinData.DSSP_SS8toSS3['-'])
                #                         print(f"- SS @ {res} of {pdbpath}")
                else:
                    print(f"Not a valid residue! Skipping: {pdbpath}  {res}")
                    continue
        # check if we lose so much residues before & after dssp, HetATM are counted
        if len(res_lst) < 50:
            print(f"Not Enough Residue Warning: {pdbpath},{len(dssp)},{len(res_lst)}")
        if len(set(ss)) <= 1:
            print(f"Not Enough SS Warning: {pdbpath},{len(dssp)},{len(res_lst)}")
        if len(model.child_list) > 1:
            print(f"Multiple Chains Warning: {pdbpath},{len(dssp)},{len(res_lst)}")
        if len(dssp) != len(res_lst):
            print(f"Missing Residue ERROR: {pdbpath},{len(dssp)},{len(res_lst)}")

        coords = np.array(coords)
        dist = squareform(pdist(coords))
        return res_lst, res_syb, ss, asa, rasa, rd, hseau, hsead, coords, dist

    @staticmethod
    def mkgraph(pdbpath, thres, local=6):
        bins = np.array([0, 0.5, *np.arange(2.5, thres + 0.1, 0.5)])
        res_lst, res_syb, ss, asa, rasa, rd, hseau, hsead, coords, distmat = ProteinData.parse_pdb(pdbpath)
        index_res = [ProteinData.AA_3LETTER.index(r) if r in ProteinData.AA_3LETTER
                     else ProteinData.AA_3LETTER.index('UNK') for r in res_syb]
        index_ss = [ProteinData.DSSP_SS3.index(s) for s in ss]
        feat_index = np.array([[r, s] for r, s in zip(index_res, index_ss)])
        feat_float = np.array([[a, b, c, d, e] for a, b, c, d, e in zip(asa, rasa, rd, hseau, hsead)])

        m_val = np.array([53.560246, 0.31603682, 2.62536, 11.12339, 16.031496])
        s_val = np.array([49.708084, 0.28938332, 1.757521, 7.9347463, 5.940793])
        feat_float = (feat_float - m_val) / s_val

        edge = np.digitize(distmat, bins) - 1
        erow_l, ecol_l, attr_l = [], [], []
        erow_r, ecol_r, attr_r = [], [], []
        for i in range(len(bins) - 1):
            er, ec = np.where(edge == i)
            lm = np.abs(er - ec) <= local
            rm = np.logical_not(lm)
            erow_l.append(er[lm])
            ecol_l.append(ec[lm])
            attr_l.append(np.ones(lm.sum()) * i)

            erow_r.append(er[rm])
            ecol_r.append(ec[rm])
            attr_r.append(np.ones(rm.sum()) * i)
        erow_l, ecol_l, attr_l = np.concatenate(erow_l), np.concatenate(ecol_l), np.concatenate(attr_l)
        erow_r, ecol_r, attr_r = np.concatenate(erow_r), np.concatenate(ecol_r), np.concatenate(attr_r)
        neighbor_idx = np.where(np.abs(erow_l - ecol_l) <= 1)[0]
        erow_r = np.concatenate([erow_r, erow_l[neighbor_idx]])
        ecol_r = np.concatenate([ecol_r, ecol_l[neighbor_idx]])
        attr_r = np.concatenate([attr_r, attr_l[neighbor_idx]])
        edge_index_l = torch.tensor([erow_l, ecol_l], dtype=torch.long)
        edge_attr_l = torch.tensor(attr_l, dtype=torch.long)
        edge_index_r = torch.tensor([erow_r, ecol_r], dtype=torch.long)
        edge_attr_r = torch.tensor(attr_r, dtype=torch.long)
        graph = Data(x=torch.tensor(feat_index, dtype=torch.long),
                     x_flt=torch.tensor(feat_float, dtype=torch.float32),
                     edge_index_l=edge_index_l, edge_attr_l=edge_attr_l,
                     edge_index_r=edge_index_r, edge_attr_r=edge_attr_r)
        return graph

    def get_pdb_path(self, pid):
        raise NotImplementedError('Subclass and implement me!')

    def convert2graph(self, names, cache_file, thres):
        if cache_file.is_file():
            graphdict = torch.load(cache_file)
        else:
            graphdict = dict()
            paths = [self.get_pdb_path(pid) for pid in names]
            res = Parallel(n_jobs=-1)(
                delayed(lambda p, t: (p, ProteinData.mkgraph(p, t)))(ppath, thres) for ppath in tqdm(paths))
            for path, graph in res:
                pid = path.name.rsplit('.', 1)[0]
                graphdict[pid] = graph
            torch.save(graphdict, cache_file)
        return graphdict


class StructureSearchDataModuleChannel(ProteinData):
    # pdbstype-2.06, error and length mismatch
    BADSET = set()  # {'d1c53a_', 'd1qcrd2', 'd2ilaa_', 'd2mysb_'}  # fail dssp

    def __init__(self, pdb_dir, tm_file, thres, pos_condition, valid_ratio, batch_size, fold=None):
        self.pdb_dir = Path(pdb_dir) if pdb_dir else None
        self.tm_file = tm_file
        self.thres = thres
        self.valid_ratio = valid_ratio
        self.pos_condition = pos_condition

        self.batch_size = batch_size
        self.fold = fold
        self.cache_file = DATA / f'{self.__class__.__name__}_graphdict_{int(thres)}.pth'

        self.eval_pdblst = None
        self.graphdict = None
        self.aligndict = None
        self.pos_pair = None
        self.no_pos = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        # we use test set some where else
        self.full_dataset = None
        super(StructureSearchDataModuleChannel, self).__init__()
        self.setup()

    def get_pdb_path(self, pid):
        return self.pdb_dir / pid[2:4] / (pid + '.ent')

    def load_pairs(self, all_pdb):
        # write a condition here!
        pos_pair = dict()
        no_pos = set()
        nameset = set(all_pdb)
        # first filter pos_candidate, for pairs only in nameset
        pos_candidate = pd.read_csv(self.tm_file).to_numpy()
        namemask = np.array([i in nameset and j in nameset for i, j, _ in pos_candidate])
        pos_candidate = pos_candidate[namemask]
        # find pairs according to each query
        for name in tqdm(all_pdb, desc='Load pairs'):
            pairs = pos_candidate[(pos_candidate[:, 0] == name) | (pos_candidate[:, 1] == name)]  # slow!!!
            if not len(pairs):
                no_pos.add(name)
                continue
            pos_mask = self.pos_condition(pairs[:, 2])
            if not pos_mask.sum():  # never be anchor
                no_pos.add(name)
                continue
            pos_pair[name] = []
            for n1, n2, s in pairs[pos_mask]:  # n1, n2 must in nameset
                if n1 == name:
                    pos_pair[name].append(n2)
                else:
                    pos_pair[name].append(n1)
        return pos_pair, no_pos

    def savetxt(self, names, fname):
        with open(DATA / (fname + '.txt'), 'w') as fout:
            for n in names:
                print(n, file=fout)

    def load_align(self):
        nlst = []
        for n1 in self.pos_pair:
            for n2 in self.pos_pair[n1]:
                nlst.append((n1, n2))
                alnfile1 = DATA / 'align' / f"{n1}-{n2}.align"
                alnfile2 = DATA / 'align' / f"{n2}-{n1}.align"
                if not alnfile1.is_file() and not alnfile2.is_file():
                    raise FileNotFoundError("alignment not found")
        aligndict = dict()
        fail = 0
        for n1, n2 in tqdm(nlst, desc="Load alignment", ncols=120):
            if not (DATA / 'align' / f"{n1}-{n2}.align").is_file():
                n1, n2 = n2, n1
            res = self.load_align_file(DATA / 'align' / f"{n1}-{n2}.align")
            if res[0] == self.graphdict[n1].x.shape[0] and res[1] == self.graphdict[n2].x.shape[0]:
                aligndict[(n1, n2)] = (res[2], res[3])
                aligndict[(n2, n1)] = (res[3], res[2])
            else:
                fail += 1
                aligndict[(n1, n2)] = ()
                aligndict[(n2, n1)] = ()
        print(f"{fail} / {len(nlst)} Failed to align, about {(fail / len(nlst)) * 100:.2f}%")
        return aligndict

    def load_align_file(self, alnfile):
        with open(alnfile) as fin:
            seq1, aln, seq2 = map(lambda x: x.rstrip("\n"), fin.readlines())
        idx1, idx2, packed = 0, 0, []
        for ii, (a1, st, a2) in enumerate(zip(seq1, aln, seq2)):
            if a1 == '-':
                idx2 += 1
                continue
            if a2 == '-':
                idx1 += 1
                continue
            idx1 += 1
            idx2 += 1
            if st in ':.':
                packed.append((a1, a2, ii, idx1 - 1, idx2 - 1))
        return idx1, idx2, \
               torch.tensor([i[3] for i in packed], dtype=torch.long), \
               torch.tensor([i[4] for i in packed], dtype=torch.long)

    def readtxt(self, fname):
        names = []
        with open(DATA / (fname + '.txt')) as fout:
            for line in fout:
                names.append(line.strip())
        return names

    def setup(self):
        # split data
        assert Path(DATA / 'full-id-cla.txt').is_file()
        name_cla_dict = dict()
        with open(Path(DATA / 'full-id-cla.txt')) as fin:
            for line in fin:
                n, c = line.split()
                name_cla_dict[n] = int(c)
        names = sorted(set(name_cla_dict.keys()) - self.BADSET)

        if self.fold is None:
            name_train, name_test = train_test_split(names, test_size=self.valid_ratio)
        else:
            if not Path(DATA / 'train-0.txt').is_file():
                kf = StratifiedKFold(n_splits=10, shuffle=True)
                names_np = np.array(names)
                for idx, (train_idx, test_idx) in enumerate(kf.split(names, [name_cla_dict[n] for n in names])):
                    self.savetxt(names_np[train_idx], f'train-{idx}')
                    self.savetxt(names_np[test_idx], f'test-{idx}')
            name_train = self.readtxt(f'train-{self.fold}')
            name_test = self.readtxt(f'test-{self.fold}')

            idx = np.random.permutation(len(name_train))
            name_train = name_train[len(name_test):]
            name_valid = name_train[:len(name_test)]

            print(f"Using Kfold split: fold {self.fold}")

        self.eval_pdblst = names
        # convert to graph. graphs are shared cross all datasets with all protein graphs
        self.graphdict = self.convert2graph(names, self.cache_file, self.thres)
        # load pos pairs:
        # for train set: pairs train vs train.
        # for others, dataset vs all (for harder query and search and ensure model didnt know them)
        pos_cache_file = DATA / f'{self.__class__.__name__}_pair.pth'
        if pos_cache_file.is_file():
            tmp = torch.load(pos_cache_file, pickle_module=dill)
            pos_pair = tmp['pos_pair']
            no_pos = tmp['no_pos']
            pos_condition = tmp['pos_condition']
            testarr = np.arange(0, 1, 0.01)
            if not np.all(pos_condition(testarr) == self.pos_condition(testarr)):
                print("Different condition used! Re-load pos pairs!")
                pos_pair, no_pos = self.load_pairs(names)
                torch.save({'pos_pair': pos_pair, 'no_pos': set(no_pos),
                            'pos_condition': self.pos_condition},
                           pos_cache_file)
        else:
            pos_pair, no_pos = self.load_pairs(names)
            no_pos = set(no_pos)
            torch.save({'pos_pair': pos_pair, 'no_pos': set(no_pos), 'pos_condition': self.pos_condition},
                       pos_cache_file)
        self.pos_pair = pos_pair
        self.no_pos = no_pos
        # load node alignment
        aln_cache_file = DATA / f'{self.__class__.__name__}_align.pth'
        if aln_cache_file.is_file():
            self.aligndict = torch.load(aln_cache_file, pickle_module=dill)
        else:
            self.aligndict = self.load_align()
            torch.save(self.aligndict, aln_cache_file)

        # split by dataset
        nameset_train = set(name_train)
        pos_pair_train = dict()
        for p, l in pos_pair.items():
            if p not in nameset_train:
                continue
            tmp = set(l) & nameset_train
            if tmp:
                pos_pair_train[p] = sorted(tmp)

        nameset_valid = set(name_valid)
        pos_pair_valid = dict()
        for p, l in pos_pair.items():
            if p in nameset_valid:
                pos_pair_valid[p] = sorted(l)

        nameset_test = set(name_test)
        pos_pair_test = dict()
        for p, l in pos_pair.items():
            if p in nameset_test:
                pos_pair_test[p] = sorted(l)

        assert len(self.graphdict) == len(self.eval_pdblst)
        assert set(self.graphdict.keys()) == set(self.eval_pdblst)
        self.train_dataset = ProteinNameDataset(sorted(name_train), pos_pair_train, self.graphdict, self.aligndict)
        self.valid_dataset = ProteinNameDataset(sorted(name_valid), pos_pair_valid, self.graphdict, self.aligndict)
        self.test_dataset = ProteinNameDataset(sorted(name_test), pos_pair_test, self.graphdict, self.aligndict)
        # we use test set some where else
        self.full_dataset = ProteinNameDataset(sorted(names), pos_pair, self.graphdict, self.aligndict)

        # misc
        nnode, nedgel, nedger = 0, 0, 0
        for k, v in self.graphdict.items():
            nnode += v.x.shape[0]
            nedgel += v.edge_index_l.shape[1]
            nedger += v.edge_index_r.shape[1]
        print(f"Nnode: {nnode}, Nedge_l: {nedgel}, Nedge_r: {nedger}, avg_edge: {(nedgel + nedger) / nnode}")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=0)


class ProteinNameDataset(Dataset):
    def __init__(self, names, pos_pair, graphdict, aligndict):
        self.names = names
        self.pos_pair = pos_pair
        self.graphdict = graphdict
        self.aligndict = aligndict
        super(ProteinNameDataset, self).__init__()

    def len(self):
        return len(self.names)

    def get(self, idx):
        return self.names[idx]

    def get_graphs(self, prots):
        graphs = [self.graphdict[prot] for prot in prots]
        graphs = Batch.from_data_list(graphs)
        return graphs

    def gen_numpy_embedding(self, model, namelst=None, device='cpu'):
        data = self if namelst is None else namelst
        loader = DataLoader(data, batch_size=128, shuffle=False, drop_last=False)
        embedding = []
        is_train = model.training
        model.eval()
        with torch.no_grad():
            for prots in loader:
                inpt = self.get_graphs(prots).to(device)
                embedding.append(model.pool(model(inpt), inpt.batch, inpt.ptr).detach().cpu().numpy())
        embedding = np.concatenate(embedding, axis=0)
        if is_train:
            model.train()
        return embedding

    @staticmethod
    def filter(a, p, n, pdblst):
        used_index = np.sort(np.unique(np.concatenate([a, p, n])))
        old2new = {o: n for n, o in enumerate(used_index)}
        new_pdblst = [pdblst[i] for i in used_index]
        new_a = np.array([old2new[i] for i in a])
        new_p = np.array([old2new[i] for i in p])
        new_n = np.array([old2new[i] for i in n])
        return (new_a, new_p, new_n), new_pdblst

    @staticmethod
    def shrink(a, p, n, pdblst, num):
        index = np.random.permutation(len(pdblst))[:num]
        old2new = {o: n for n, o in enumerate(index)}
        new_pdblst = [pdblst[i] for i in index]
        new_a, new_p, new_n = [], [], []
        for ai, pi, ni in zip(a, p, n):
            if ai in index and pi in index and ni in index:
                new_a.append(old2new[ai])
                new_p.append(old2new[pi])
                new_n.append(old2new[ni])
        new_a = np.array(new_a)
        new_p = np.array(new_p)
        new_n = np.array(new_n)
        return (new_a, new_p, new_n), new_pdblst

    # this is only called on train set!
    @torch.no_grad()
    def sample_batch_triplet(self, model, queries, margin, hardpos=False, hardneg=False, hardest=False,
                             k=1, sample_from=0.20, device='cpu', max=None, mult=1):
        # find if there are any positive pairs in it
        anchorlst = queries
        add, allemb, namelst, tmp_name2idx, tmp_dist = list(), None, None, None, None
        for query in queries:
            if query not in self.pos_pair:
                continue
            pos_candidate_lst = self.pos_pair[query]
            pos_candidate_set = set(pos_candidate_lst)
            if int(hardpos) + int(hardneg) + int(hardest) > 1:
                raise ValueError("hardpos, hardneg and hardest can only have one True!")
            chosen = []
            if hardpos:
                pos_emb = self.gen_numpy_embedding(model, [query] + pos_candidate_lst, device)
                pos_dist = cdist(pos_emb[None, 0, :], pos_emb[1:]).squeeze()
                index = pos_dist.argsort()[-k:]  # farthest positive
                chosen = [pos_candidate_lst[i] for i in index]
            else:
                if namelst is None:
                    assert allemb is None and tmp_name2idx is None and tmp_dist is None
                    candidates = set()
                    for q in queries:
                        if q in self.pos_pair:
                            candidates.update(self.pos_pair[q])
                    candidates.update(queries)
                    others_candidate = sorted(set(self.names) - candidates)
                    others = random.sample(others_candidate,
                                           k=min(len(others_candidate), int(sample_from * len(self.graphdict))))
                    namelst = sorted(candidates) + others
                    tmp_name2idx = {n: i for i, n in enumerate(namelst)}
                    allemb = self.gen_numpy_embedding(model, namelst, device)
                    tmp_dist = squareform(pdist(allemb))
                query_idx = tmp_name2idx[query]
                pos_idx = [idx for idx, n in enumerate(namelst) if n in pos_candidate_set]
                neg_idx = [idx for idx, n in enumerate(namelst) if n not in pos_candidate_set and n != query]
                pos_dist = tmp_dist[query_idx, pos_idx]
                neg_dist = tmp_dist[query_idx, neg_idx]
                if hardneg:  # this need at least one positive!
                    chosen = [i for i in np.random.choice(pos_candidate_lst, int(k * mult))]
                    chosen += [namelst[i] for i in neg_dist.argsort()[:k]]  # closest negative
                elif hardest:
                    chosen = [namelst[i] for i in pos_dist.argsort()[-k:]]  # farthest positive
                    chosen += [namelst[i] for i in neg_dist.argsort()[:k]]  # closest negative
            add += chosen
        # resulting all protein in batch have k positive pair
        # may be repeat names
        queries = sorted(set(queries + add))
        queries_emb = self.gen_numpy_embedding(model, queries, device)
        queries_dist = squareform(pdist(queries_emb))
        # sample triplets
        batch_pdblst = queries
        batch_name2idx = {pdb: idx for idx, pdb in enumerate(batch_pdblst)}
        batch_a, batch_p, batch_n = [], [], []

        allowed = [a for a in anchorlst if a in self.pos_pair]
        for anchor in allowed:
            pos_candidate_set = set(self.pos_pair[anchor])
            anchor_idx = batch_name2idx[anchor]
            pos_index = np.array([idx for idx, i in enumerate(batch_pdblst) if i in pos_candidate_set])
            neg_index = np.array(
                [idx for idx, i in enumerate(batch_pdblst) if i not in pos_candidate_set and i != anchor])
            for pidx, nidx in itertools.product(pos_index, neg_index):
                if queries_dist[anchor_idx, pidx] - queries_dist[anchor_idx, nidx] + margin <= 0:  # zero loss
                    continue
                batch_a.append(anchor_idx)
                batch_p.append(pidx)
                batch_n.append(nidx)

        (batch_a, batch_p, batch_n), batch_pdblst = self.filter(batch_a, batch_p, batch_n, batch_pdblst)
        if max is not None and len(batch_pdblst) > max:
            (batch_a, batch_p, batch_n), batch_pdblst = self.shrink(batch_a, batch_p, batch_n, batch_pdblst, max)
        return (batch_a, batch_p, batch_n), batch_pdblst

    @staticmethod
    def calc_metrics(gt, score):
        metrics = [roc_auc_score(gt, score), average_precision_score(gt, score)]
        hitk, prec, recl = metrics_at_k(gt, score, [1, 5, 10, 20, 50, 100])
        metrics += hitk + prec + recl
        return metrics

    @staticmethod
    def evaluate(pos_pair: dict, names_q: list, embed_q: np.ndarray, names_t: list, embed_t: np.ndarray, metric: str):
        # assert names_q all in pos_pair.keys()
        # assert names_t all in pos_pair.values()
        scoremat = 1 / (1e-6 + cdist(embed_q, embed_t, metric=metric))
        metrics = []
        for idx, query in enumerate(names_q):
            if query not in pos_pair:
                continue
            posset = set(pos_pair[query])
            gt = np.array([(n in posset or n == query) for n in names_t], dtype=np.float)
            score = scoremat[idx]
            mask = np.ones(len(names_t), dtype=np.bool)
            try:
                index = names_t.index(query)
            except ValueError as e:
                index = -1
            if index >= 0:
                mask[index] = 0

            gt, score = gt[mask], score[mask]
            metrics.append(ProteinNameDataset.calc_metrics(gt, score))
        return np.array(metrics)

    def evaluate_train(self, model, device, metric):
        embed_train = self.gen_numpy_embedding(model, device=device)
        return self.evaluate(self.pos_pair, self.names, embed_train, self.names, embed_train, metric)

    def evaluate_val(self, model, device, metric):
        if len(self.graphdict) == len(self.names):  # full set
            embed_t = embed_q = self.gen_numpy_embedding(model, device=device)
            target_lst = self.names
        else:
            embed_q = self.gen_numpy_embedding(model, device=device)
            target_lst = sorted(self.graphdict.keys())
            embed_t = self.gen_numpy_embedding(model, namelst=target_lst, device=device)
        return self.evaluate(self.pos_pair, self.names, embed_q, target_lst, embed_t, metric)
