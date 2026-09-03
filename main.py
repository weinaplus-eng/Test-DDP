import argparse
import os

import torch
from Utils.distributed import cleanup_distributed, init_distributed, is_main
from Utils.load_Data import load_data
from Utils.load_Vict import load_GNN


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--cuda', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--Vit_model', type=str, default='GCN', choices=['GIN', 'GCN', 'GAT', 'ARMA'])
    parser.add_argument('--save_root', type=str, default='Results')
    parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='set by torch.distributed.launch, torchrun uses LOCAL_RANK instead',
    )
    parser.add_argument(
        '--dist_backend',
        type=str,
        default=None,
        choices=['nccl', 'gloo'],
        help='defaults to nccl on GPU and gloo on CPU',
    )

    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--heads', type=int, default=1)
    parser.add_argument('--arma_stacks', type=int, default=2)
    parser.add_argument('--arma_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=100)

    args = parser.parse_args()
    args.use_cuda = args.cuda == 'cuda' and torch.cuda.is_available()
    return args


@torch.no_grad()
def predict(model, data, device):
    model.eval()
    data = data.to(device)
    out = model(data.x, data.edge_index)
    logits = out[0] if isinstance(out, tuple) else out
    pred = logits.argmax(dim=-1)

    def acc(mask):
        if mask is None or int(mask.sum()) == 0:
            return float('nan')
        return float((pred[mask] == data.y[mask]).float().mean().item())

    return {
        'pred': pred,
        'train_acc': acc(data.train_mask),
        'val_acc': acc(data.val_mask),
        'test_acc': acc(data.test_mask),
    }


def main():
    args = get_args()
    init_distributed(args)
    try:
        if is_main(args):
            os.makedirs(args.save_root, exist_ok=True)

        dataset, data_inf = load_data(
            args.dataset,
            args.seed,
            rank=args.rank,
            world_size=args.world_size,
        )
        model = load_GNN(args, dataset, data_inf)
        report = predict(model, dataset, args.device)

        if is_main(args):
            print('========== prediction ==========')
            print(f'{args.dataset}/{args.Vit_model}')
            print(
                f'train {report["train_acc"]:.4f}  '
                f'val {report["val_acc"]:.4f}  '
                f'test {report["test_acc"]:.4f}'
            )
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
