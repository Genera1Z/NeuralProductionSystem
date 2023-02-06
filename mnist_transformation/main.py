import os
import os.path as osp
import random

import numpy as np
import torch as pt
import torch.backends.cudnn as ptbc
import torch.nn as nn
import torch.optim as pto
import torch.utils.data as ptud
import torchvision.transforms as ptvt
from einops import rearrange
from termcolor import colored
from tqdm import tqdm

from datum import MnistOperationSet
from model import MnistOperationNps


def count_mapping(s: np.ndarray, t: np.ndarray, keys, return_string=True):
    assert len(s.shape) == len(t.shape) == 1
    assert len(s) == len(t)
    # assert s.dtype in (np.uint8, np.uint16, np.uint32, np.uint64)
    # assert t.dtype in (np.uint8, np.uint16, np.uint32, np.uint64)
    func = lambda _, a, b: dict(zip(
        *np.unique(b[np.where(a == _)[0]], return_counts=True)
    ))
    cnts_t = []
    for idx_s, key_s in enumerate(keys):
        cnt_t = func(idx_s, s, t)
        cnts_t.append(cnt_t)
    map_st = dict(zip(keys, cnts_t))
    if return_string:
        map_st = str(map_st).replace("'", '')
    return map_st


def epoch_t(model, dataset, metric, optim, device, n_rule):
    model.train()

    losses = []
    rule_golden = []
    rule_actual = []
    var_actual = []

    for i, batch in tqdm(enumerate(dataset)):
        frames = batch[0].to(device)
        operations = batch[1].to(device)

        inputs_ = frames[:, :-1, ...].flatten(0, 1)[:, None, ...]  # TODO (b*t,2,c,h,w)
        operations_ = operations.flatten(0, 1)[:, None, ...]

        model.rule_network.reset()
        optim.zero_grad()

        outputs_ = model(inputs_, operations_)
        targets_ = frames[:, 1:, ...].flatten(0, 1)  # TODO inputs_[:, 1, :, :]
        loss = metric(outputs_[:, 0, ...], targets_)
        losses.append(loss.item())  # loss.detach().cpu().item()

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # XXX remove?
        optim.step()

        rule_golden.append(operations_.detach()[:, 0, ...].argmax(dim=1).cpu())
        rule_actual.append(np.array(model.rule_network.rule_selection))
        var_actual.append(np.array(model.rule_network.primary_selection))

    loss = np.mean(losses)

    rule_golden = np.concatenate(rule_golden, axis=0)
    rule_actual = rearrange(rule_actual, 'n t b -> (n b t)')
    var_actual = rearrange(var_actual, 'n t b -> (n b t)')

    map_rule_ga = count_mapping(rule_golden, rule_actual, range(n_rule))
    map_rule_var = count_mapping(rule_actual, var_actual, range(n_rule))

    return loss, map_rule_ga, map_rule_var


@pt.no_grad()
def epoch_v(model, dataset, metric, device, n_rule):
    model.eval()

    losses = []
    rule_golden = []
    rule_actual = []
    var_actual = []

    for i, batch in tqdm(enumerate(dataset)):
        frames = batch[0].to(device)
        operations = batch[1].to(device)

        inputs = frames[:, :-1, ...]

        model.rule_network.reset()
        outputs = model(inputs, operations)
        loss = metric(outputs.flatten(0, 1), frames[:, 1:, ...].flatten(0, 1))
        losses.append(loss.item())

        rule_golden.append(operations.flatten(0, 1).argmax(dim=1).cpu())
        rule_actual.append(np.array(model.rule_network.rule_selection))
        var_actual.append(np.array(model.rule_network.primary_selection))

    loss = np.mean(losses)

    rule_golden = np.concatenate(rule_golden, axis=0)
    rule_actual = rearrange(rule_actual, 'n t b -> (n b t)')
    var_actual = rearrange(var_actual, 'n t b -> (n b t)')

    map_op_rule = count_mapping(rule_golden, rule_actual, range(n_rule))
    map_rule_var = count_mapping(rule_actual, var_actual, range(n_rule))

    return loss, map_op_rule, map_rule_var


def main():
    n_epoch = 100
    batch_size = 64
    lr0 = 0.0001
    n_op = 4
    n_rule = 4
    c_rule = 6
    n_var = 2
    c_var = 128
    device = pt.device('cuda')

    dload_t = ptud.DataLoader(
        MnistOperationSet(n_op, '../data/', train=True, transform=[ptvt.ToTensor()]),
        batch_size, True, num_workers=4, collate_fn=None, drop_last=True
    )
    dload_v = ptud.DataLoader(
        MnistOperationSet(n_op, '../data/', train=False, transform=[ptvt.ToTensor()]),
        batch_size, False, num_workers=4, collate_fn=None, drop_last=True
    )
    model = MnistOperationNps(n_rule, c_rule, n_var, c_var).to(device)
    metric = nn.BCELoss()
    optim = pto.Adam(model.parameters(), lr=lr0)

    ####
    # frames, operations = dload_t.dataset.__getitem__(0)
    # frames = frames[None, ...][:, :-1]
    # operations = operations[None, ...]
    # model.train()
    # outputs = model(frames, operations)
    ####

    ckpt_path = './res/'
    best_ckpt_file = osp.join(ckpt_path, f'best.pth')
    if not osp.exists(ckpt_path):
        os.makedirs(ckpt_path)

    loss_min = 10.
    for n in range(n_epoch):
        loss_v, map_op_rule_v, map_rule_var_v = epoch_v(model, dload_v, metric, device, n_rule)
        print(colored(
            f'[val epoch{n:04d}]\tloss={loss_v:.10f}\n{map_op_rule_v}\n{map_rule_var_v}',
            'green'))
        loss_t, map_op_rule_t, map_rule_var_t = epoch_t(model, dload_t, metric, optim, device, n_rule)
        print(
            f'[train epoch{n:04d}]\tloss={loss_t:.10f}\n{map_op_rule_t}\n{map_rule_var_t}'
        )
        if loss_v < loss_min:
            loss_min = loss_v
            print('...saving new best model checkpoint...')
            pt.save(model, best_ckpt_file)


def set_seed(seed):
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # pt.use_deterministic_algorithms(True, warn_only=True)
    print(f'seed={seed}')
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    pt.manual_seed(seed)
    # pt.cuda.manual_seed(seed)
    pt.cuda.manual_seed_all(seed)
    # ptbc.benchmark = False
    ptbc.deterministic = True


if __name__ == '__main__':
    # XXX 原版实现在seed=0时可以，不设seed时表现也很不稳定，时常只能激活俩规则。
    # set_seed(1)
    main()
