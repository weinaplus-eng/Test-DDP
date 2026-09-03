import json
import os
import random

import torch
import torch.nn.functional as F

from GNN import ARMA, GAT, GCN, GIN
from Utils.distributed import barrier, broadcast_flag, is_main, unwrap_model, wrap_ddp


def _set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _split_metrics(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    logits = out[0] if isinstance(out, tuple) else out

    y_true = data.y[mask]
    y_pred = logits[mask].argmax(dim=-1)
    acc = float((y_pred == y_true).float().mean().item())
    loss = float(F.cross_entropy(logits[mask], y_true).item())
    return acc, loss


def _evaluate(model, data):
    train_acc, train_loss = _split_metrics(model, data, data.train_mask)
    val_acc, val_loss = _split_metrics(model, data, data.val_mask)
    test_acc, test_loss = _split_metrics(model, data, data.test_mask)
    return {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'test_loss': test_loss,
    }


def _save_results(ckpt_path, model, metrics, best_epoch):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    raw = unwrap_model(model)
    torch.save({'state_dict': raw.state_dict(), 'metrics': metrics, 'best_epoch': best_epoch}, ckpt_path)
    record = {'best_epoch': best_epoch, **{k: round(v, 6) for k, v in metrics.items()}}
    with open(ckpt_path.replace('.pt', '.json'), 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=2)


def _local_train_idx(data, args):
    idx = data.train_mask.nonzero(as_tuple=False).view(-1)
    world_size = int(getattr(args, 'world_size', 1))
    rank = int(getattr(args, 'rank', 0))
    if world_size <= 1:
        return idx
    n = idx.numel()
    chunk = (n + world_size - 1) // world_size
    start = min(rank * chunk, n)
    end = min(start + chunk, n)
    return idx[start:end]


def _train(model, data, args):
    lr = getattr(args, 'lr', 0.01)
    weight_decay = getattr(args, 'weight_decay', 5e-4)
    epochs = getattr(args, 'epochs', 200)
    patience = getattr(args, 'patience', 50)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_idx = _local_train_idx(data, args)
    world_size = int(getattr(args, 'world_size', 1))
    n_total = int(data.train_mask.sum().item())
    n_local = int(train_idx.numel())

    best_val_acc = -1.0
    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None
    stall = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        logits = out[0] if isinstance(out, tuple) else out
        if n_local == 0:
            loss = logits.sum() * 0.0
        else:
            loss = F.cross_entropy(logits[train_idx], data.y[train_idx])
            if world_size > 1 and n_total > 0:
                loss = loss * (world_size * n_local / n_total)
        loss.backward()
        optimizer.step()

        metrics = _evaluate(unwrap_model(model), data)
        improved = (
            metrics['val_acc'] > best_val_acc
            or (metrics['val_acc'] == best_val_acc and metrics['val_loss'] < best_val_loss)
        )
        if improved:
            best_val_acc = metrics['val_acc']
            best_val_loss = metrics['val_loss']
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in unwrap_model(model).state_dict().items()}
            stall = 0
        else:
            stall += 1

        if is_main(args) and epoch == 1:
            print(f'{"epoch":>6}  {"loss":>8}  {"train":>8}  {"val":>8}  {"test":>8}')
        stop = stall >= patience
        if is_main(args) and (epoch == 1 or epoch % 20 == 0 or stop or epoch == epochs):
            flag = ' *' if epoch == best_epoch else ''
            print(
                f'{epoch:6d}  {loss.item():8.4f}  '
                f'{metrics["train_acc"]:8.4f}  {metrics["val_acc"]:8.4f}  '
                f'{metrics["test_acc"]:8.4f}{flag}'
            )
        if stop:
            break

    if best_state is not None:
        unwrap_model(model).load_state_dict(best_state)
    return best_epoch


def load_GNN(args, dataset, data_inf):
    root = os.path.join(args.save_root, 'GNN')
    if is_main(args):
        os.makedirs(root, exist_ok=True)
    barrier()
    extra = {
        'GAT': f'_heads{args.heads}',
        'ARMA': f'_stacks{args.arma_stacks}_layers{args.arma_layers}',
    }.get(args.Vit_model, '')
    ckpt_path = os.path.join(
        root,
        f'{args.dataset}_{args.Vit_model}_seed{args.seed}_L{args.layers}_H{args.hidden_dim}_d{args.dropout}{extra}.pt',
    )

    _set_seed(args.seed)
    data = dataset.to(args.device)

    model_cls = {'GCN': GCN, 'GIN': GIN, 'GAT': GAT, 'ARMA': ARMA}[args.Vit_model]
    model = model_cls(args, data_inf['num_features'], data_inf['num_classes']).to(args.device)

    has_ckpt = is_main(args) and os.path.isfile(ckpt_path)
    has_ckpt = broadcast_flag(has_ckpt, args)

    if has_ckpt:
        if is_main(args) or os.path.isfile(ckpt_path):
            if is_main(args):
                print(f'load  {args.dataset}/{args.Vit_model}  {os.path.basename(ckpt_path)}')
            ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
            state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
            model.load_state_dict(state)
        model = wrap_ddp(model, args)
        metrics = _evaluate(unwrap_model(model), data)
        if is_main(args):
            print(
                f'      train {metrics["train_acc"]:.4f}  val {metrics["val_acc"]:.4f}  '
                f'test {metrics["test_acc"]:.4f}'
            )
        return unwrap_model(model)

    if is_main(args):
        print(f'train {args.dataset}/{args.Vit_model}')
    model = wrap_ddp(model, args)
    best_epoch = _train(model, data, args)
    metrics = _evaluate(unwrap_model(model), data)
    if is_main(args):
        _save_results(ckpt_path, model, metrics, best_epoch)
        print(
            f'save  {os.path.basename(ckpt_path)}  (best epoch {best_epoch})\n'
            f'      train {metrics["train_acc"]:.4f}  val {metrics["val_acc"]:.4f}  '
            f'test {metrics["test_acc"]:.4f}'
        )
    barrier()
    return unwrap_model(model)
