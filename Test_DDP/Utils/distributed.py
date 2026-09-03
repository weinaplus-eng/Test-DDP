import os
import socket
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def init_distributed(args):
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank_env = os.environ.get('LOCAL_RANK')
    if local_rank_env is not None:
        local_rank = int(local_rank_env)
    else:
        local_rank = int(getattr(args, 'local_rank', 0) or 0)
        if local_rank < 0:
            local_rank = 0

    args.world_size = world_size
    args.rank = rank
    args.local_rank = local_rank
    args.distributed = world_size > 1
    use_cuda = bool(getattr(args, 'use_cuda', False))

    if not args.distributed:
        args.device = torch.device('cuda' if use_cuda else 'cpu')
        args.dist_backend = None
        return args

    if use_cuda:
        torch.cuda.set_device(local_rank)
        args.device = torch.device('cuda', local_rank)
        backend = args.dist_backend or 'nccl'
    else:
        args.device = torch.device('cpu')
        backend = args.dist_backend or 'gloo'

    dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
    args.dist_backend = dist.get_backend()

    print(
        f'[DDP] rank {rank}/{world_size}  local_rank={local_rank}  '
        f'host={socket.gethostname()}  device={args.device}',
        flush=True,
    )
    dist.barrier()
    if rank == 0:
        print(
            f'[DDP] all {world_size} processes ready  backend={args.dist_backend}',
            flush=True,
        )
    return args


def wrap_ddp(model, args):
    if not getattr(args, 'distributed', False):
        return model
    if args.device.type == 'cuda':
        return DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
        )
    return DDP(model)


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def is_main(args=None):
    if args is not None:
        return int(getattr(args, 'rank', 0)) == 0
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def broadcast_flag(flag, args):
    tensor = torch.tensor([int(bool(flag))], device=args.device)
    if getattr(args, 'distributed', False) and dist.is_initialized():
        dist.broadcast(tensor, src=0)
    return bool(tensor.item())


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
