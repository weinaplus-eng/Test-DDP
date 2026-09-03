import os
import ssl
import urllib.request

import torch
import torch.distributed as dist
from torch_geometric.datasets import Amazon, CitationFull, Coauthor, Planetoid
from torch_geometric.utils import remove_self_loops, to_undirected

_PLANETOID_MIRRORS = (
    'https://github.com/kimiyoung/planetoid/raw/master/data',
    'https://raw.githubusercontent.com/kimiyoung/planetoid/master/data',
    'https://cdn.jsdelivr.net/gh/kimiyoung/planetoid@master/data',
)


def _split_masks(y, seed, n_train=20, n_val=30):
    #没有官方划分时，按 seed 做每类 20/30/rest 分层抽样
    g = torch.Generator().manual_seed(int(seed))
    n = y.size(0)
    train = torch.zeros(n, dtype=torch.bool)
    val = torch.zeros(n, dtype=torch.bool)
    test = torch.zeros(n, dtype=torch.bool)
    for c in y.unique():
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        idx = idx[torch.randperm(idx.numel(), generator=g)]
        k = idx.numel()
        n_tr, n_va = n_train, n_val
        if k < n_train + n_val + 1:
            n_tr = max(1, k // 3)
            n_va = max(1, (k - n_tr) // 2)
        train[idx[:n_tr]] = True
        val[idx[n_tr:n_tr + n_va]] = True
        test[idx[n_tr + n_va:]] = True
    return train, val, test


def _sync_dataset(rank, world_size):
    if world_size <= 1 or not dist.is_available() or not dist.is_initialized():
        return
    if rank != 0:
        dist.barrier()


def _release_dataset(rank, world_size):
    if world_size <= 1 or not dist.is_available() or not dist.is_initialized():
        return
    if rank == 0:
        dist.barrier()


def load_data(data_name, seed, data_root='Dataset', rank=0, world_size=1):
    _sync_dataset(rank, world_size)
    name = data_name.strip().lower().replace('-', '_')
    root = data_root if os.path.isabs(data_root) else os.path.join('.', data_root)

    if name in ('cora', 'citeseer', 'pubmed'):
        pyg_name = {'cora': 'Cora', 'citeseer': 'CiteSeer', 'pubmed': 'PubMed'}[name]
        _prepare_planetoid(root, pyg_name)
        dataset = Planetoid(root=root, name=pyg_name)
    elif name in ('cora_ml', 'coraml'):
        dataset = CitationFull(root=root, name='Cora_ML')
    elif name in ('computers', 'amazon_computers'):
        dataset = Amazon(root=root, name='Computers')
    elif name in ('photo', 'amazon_photo'):
        dataset = Amazon(root=root, name='Photo')
    elif name in ('cs', 'coauthor_cs'):
        dataset = Coauthor(root=root, name='CS')
    elif name in ('physics', 'coauthor_physics'):
        dataset = Coauthor(root=root, name='Physics')
    else:
        raise ValueError(
            f'Unknown dataset {data_name}. '
            'Use Cora, CiteSeer, PubMed, Cora_ML, Computers, Photo, CS, or Physics.'
        )

    data = dataset[0]
    data.x = data.x.float()
    data.y = data.y.view(-1).long()
    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    data.edge_index, _ = remove_self_loops(data.edge_index)

    if getattr(data, 'train_mask', None) is None:
        data.train_mask, data.val_mask, data.test_mask = _split_masks(data.y, seed)

    data_inf = {
        'num_features': int(data.x.size(1)),
        'num_classes': int(int(data.y.max().item()) + 1),
        'num_nodes': int(data.num_nodes),
    }
    if rank == 0:
        print(
            f'{data_name}  n={data.num_nodes}  '
            f'train {int(data.train_mask.sum())}  val {int(data.val_mask.sum())}  '
            f'test {int(data.test_mask.sum())}'
        )
    _release_dataset(rank, world_size)
    return data, data_inf


def _ssl_contexts():
    ctxs = []
    try:
        import certifi
        ctxs.append(ssl.create_default_context(cafile=certifi.where()))
    except Exception:
        pass
    ctxs.append(ssl._create_unverified_context())
    return ctxs


def _download(urls, path):
    #绕开 PyG/fsspec 的 GitHub SSL 校验失败
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    last = None
    for url in urls:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        for ctx in _ssl_contexts():
            try:
                with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
                    data = r.read()
                if not data:
                    continue
                tmp = path + '.part'
                with open(tmp, 'wb') as f:
                    f.write(data)
                os.replace(tmp, path)
                return
            except Exception as e:
                last = e
    raise RuntimeError(f'cannot download {os.path.basename(path)}') from last


def _prepare_planetoid(root, pyg_name):
    raw_dir = os.path.join(root, pyg_name, 'raw')
    for part in ('x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index'):
        fname = f'ind.{pyg_name.lower()}.{part}'
        urls = [f'{base}/{fname}' for base in _PLANETOID_MIRRORS]
        _download(urls, os.path.join(raw_dir, fname))
