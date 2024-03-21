import os
import argparse
from pathlib import Path

import torch
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from abcdict import AbcDict

RANK = int(os.environ['RANK'])
LOCAL_RANK = int(os.environ['LOCAL_RANK'])
WORLD_SIZE = int(os.environ['WORLD_SIZE'])
DEVICE = torch.device('cuda', LOCAL_RANK)
distributed.init_process_group('nccl')
R0 = RANK == 0


def main(args):
    cfg_file = args.config
    if R0:
        assert Path(cfg_file).exists(), f'can not find config file: {cfg_file}'
    cfg = AbcDict(cfg_file)
    if R0:
        print(f'config file is: {cfg_file}')
        print(cfg)

    train_set = None
    train_sampler = DistributedSampler(train_set)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.train.batch_size,
        sampler=train_sampler
    )

    model = None
    model = DistributedDataParallel(
        model.cuda(DEVICE),
        device_ids=[LOCAL_RANK],
        output_device=LOCAL_RANK
    )

    optimizer = None
    if cfg.train.optim.name == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            cfg.train.optim.lr,
            cfg.train.optim.momentum
        )
    elif cfg.train.optim.name == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.train.optim.lr,
            weight_decay=cfg.train.optim.weight_decay
        )
    assert optimizer, f'can not find optim in config file: {cfg_file}'

    for epoch in range(cfg.train.epoch_num):
        for batch_idx, item in enumerate(train_loader):
            data, label = item
            optimizer.zero_grad()
            output = model(data)
            loss = None
            loss.backward()
            optimizer.step()
            if R0:
                print(loss)
                torch.save(model.module, './model.pth')
                # model.to(DEVICE)


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description='train'
    )
    parser.add_argument(
        '-cfg', '--config', type=str, help='yaml config file',
        default=str(Path(__file__).resolve().parent / 'config/cfg.yaml')
    )
    main(parser.parse_args())
