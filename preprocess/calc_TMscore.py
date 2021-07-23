import argparse
import logging
import time
from itertools import combinations
from pathlib import Path
from subprocess import run, TimeoutExpired

from joblib import Parallel, delayed
from tqdm import tqdm

# we must use fortran version of TMalign
parser = argparse.ArgumentParser()
parser.add_argument('tmalign')
parser.add_argument('pdb_dir')
parser.add_argument('--cache', dest='cache', default=None, type=str)
parser.add_argument('--chunk', dest='chunk', default=10000, type=int)
parser.add_argument('--njobs', dest='njobs', default=-1, type=int)

args = parser.parse_args()
tmalign_path = Path(args.tmalign)
pdb_dir = Path(args.pdb_dir)

cache_file = Path(args.cache) if args.cache is not None else None
chunk = args.chunk
njobs = args.njobs


def start_logger_if_necessary():
    global pdb_dir
    logger = logging.getLogger("TMalign-logger")
    if len(logger.handlers) == 0:
        logger.setLevel(logging.INFO)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
        fh = logging.FileHandler(pdb_dir / 'tmalign.log', mode='w')
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
        logger.addHandler(sh)
        logger.addHandler(fh)
    return logger


def evaluate(pdb1, pdb2, tmalign, timeout=600):
    logger = start_logger_if_necessary()
    try:
        result = run([str(tmalign), str(pdb1), str(pdb2), '-a'], capture_output=True, timeout=timeout, encoding='utf-8')
    except TimeoutExpired:
        logger.warning(f"{str(pdb1)}, {str(pdb2)}, Running time exceed {timeout / 60} min, killed!")
        return (pdb1.stem, pdb2.stem), None
    if result.returncode != 0:
        logger.warning(f"{str(pdb1)}, {str(pdb2)}, Something wrong, return not zero.")
        return (pdb1.stem, pdb2.stem), None
    else:
        s = result.stdout.split('\n')
        score = (float(s[17][10:17]), float(s[18][10:17]), float(s[19][10:17]))
        return (pdb1.stem, pdb2.stem), score


def write(chunk_score, file_handle):
    cnt = 0
    for pdbs, score in chunk_score:
        if score is None:
            continue
        print(f"{pdbs[0]}\t{pdbs[1]}\t{score[0]}\t{score[1]}\t{score[2]}", file=file_handle)
        cnt += 1
    file_handle.flush()
    return cnt


def half_result(fname):
    cache, length = dict(), 0
    with open(fname) as fin:
        for line in fin:
            if line.startswith('#'):
                continue
            elem = line.split('\t')
            try:
                assert len(elem) == 5
                key = (elem[0], elem[1])
                val = (float(elem[2]), float(elem[3]), float(elem[4]))
                assert len(elem[0]) == len(elem[1]) == 5
                cache[key] = val
                length += 1
            except Exception as e:
                break
    print(f"Cached {length} computed results.")
    return cache


if __name__ == "__main__":
    pdblst = list(pdb_dir.glob('**/*.ent'))
    save_path = pdb_dir / 'tmalign.tsv'
    logger = start_logger_if_necessary()

    # load half result
    if save_path.is_file():
        cache = half_result(save_path)
    else:
        cache = dict()
    # load 3rd cache
    if cache_file is not None:
        tmp = half_result(cache_file)
        for k, v in tmp.items():
            n1, n2 = k
            if (n1, n2) in cache or (n2, n1) in cache:
                continue
            cache[k] = v

    outfile = open(save_path, 'w')

    ideal_count = len(pdblst) * (len(pdblst) - 1) / 2
    count, last_idx = 0, 0
    print(f"Total pairs: {ideal_count / 1000}k")
    print('#PDBID1\t#PDBID2\t#SCORE1\t#SCORE2\t#SCORE_AVG', file=outfile)
    joblst, use_cache, time_record = [], [], []
    seen = set()

    for idx, (pdb1, pdb2) in tqdm(enumerate(combinations(pdblst, 2), 1), total=ideal_count):
        # check if this pair is computed before
        if (pdb1.stem, pdb2.stem) in cache:
            use_cache.append(((pdb1.stem, pdb2.stem), cache[(pdb1.stem, pdb2.stem)]))
            seen.add((pdb1.stem, pdb2.stem))
            cache.pop((pdb1.stem, pdb2.stem), None)
            continue
        elif (pdb2.stem, pdb1.stem) in cache:
            use_cache.append(((pdb2.stem, pdb1.stem), cache[(pdb2.stem, pdb1.stem)]))
            seen.add((pdb2.stem, pdb1.stem))
            cache.pop((pdb2.stem, pdb1.stem), None)
            continue

    ideal_count = ideal_count - len(seen)
    if len(use_cache) > 0:
        print(f"Used {len(use_cache) / 1000}k cached records, total need to compute {ideal_count}")
        count += write(use_cache, outfile)
    del use_cache
    del cache

    pbar = tqdm(desc="Calculating", total=len(pdblst) * (len(pdblst) - 1) / 2)
    for idx, (pdb1, pdb2) in enumerate(combinations(pdblst, 2), 1):
        pbar.update(1)
        last_idx = idx
        if (pdb1.stem, pdb2.stem) in seen or (pdb2.stem, pdb1.stem) in seen:
            continue
        # add to job list
        joblst.append((pdb1, pdb2))
        if len(joblst) < chunk:
            continue
        else:
            start_time = time.time()
            chunk_score = Parallel(n_jobs=njobs)(delayed(evaluate)(pdb1, pdb2, tmalign_path) for pdb1, pdb2 in joblst)
            count += write(chunk_score, outfile)
            end_time = time.time()
            time_record.append(end_time - start_time)
            joblst.clear()
    pbar.close()
    # the last ones, save life, just compute them
    print(f'# {(last_idx - len(joblst)) / 1000}k - {last_idx / 1000}k')
    chunk_score = Parallel(n_jobs=njobs)(delayed(evaluate)(pdb1, pdb2, tmalign_path) for pdb1, pdb2 in joblst)
    count += write(chunk_score, outfile)

    # stats
    logger.info(f"Total record pair: {count / 1000}k / {len(pdblst) * (len(pdblst) - 1) / 2 / 1000}k")
    logger.info(f"Saved to {outfile.name}")
