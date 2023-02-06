import os
import os.path as osp
import random

import numpy as np
import torch as pt
import torch.backends.cudnn as ptbc
import torch.nn as ptn
import torch.optim as pto
import torch.utils.data as ptud
from einops import rearrange
from termcolor import colored
from tqdm import tqdm

from datum import CoordinateArithmeticSet
from model import CoordinateArithmeticNps


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


def epoch_t(model, dataset, metric, optim, device, n_slot, n_rule):
    model.train()

    losses = []
    rule_golden = []
    rule_actual = []
    var_golden = []
    var_actual = []

    for i, batch in tqdm(enumerate(dataset)):
        frames = batch[0].to(device)  # (b,t,c,d)
        operations = batch[1].numpy()  # (b,t,3)

        model.rule_network.reset()
        optim.zero_grad()

        inputs = rearrange([frames[:, :-1, :, :], frames[:, 1:, :, :]], 'n b t c d -> (b t) n c d', n=2)  # (b*t,2,c,d)
        outputs = model(inputs)  # (b*t,1,c,d)
        loss = metric(outputs[:, 0, :, :], inputs[:, 1, :, :])
        losses.append(loss.item())

        loss.backward()
        optim.step()

        rule_golden.append(operations[:, :, -1])
        rule_actual.append(np.array(model.rule_network.rule_selection))
        var_golden.append(operations[:, :, 0])
        var_actual.append(np.array(model.rule_network.primary_selection))

    train_loss = np.mean(losses)

    rule_golden = rearrange(rule_golden, 'n b t-> (n b t)')
    rule_actual = rearrange(rule_actual, 'n t b -> (n b t)')
    var_golden = rearrange(var_golden, 'n b t -> (n b t)')
    var_actual = rearrange(var_actual, 'n t b -> (n b t)')

    map_rule_ga = count_mapping(rule_golden, rule_actual, range(n_rule))
    map_var_ga = count_mapping(var_golden, var_actual, range(n_slot))

    return train_loss, map_rule_ga, map_var_ga


@pt.no_grad()
def epoch_v(model, dataset, metric, device, n_slot, n_rule):
    model.eval()

    losses = []
    rule_golden = []
    rule_actual = []
    var_golden = []
    var_actual = []

    for i, batch in tqdm(enumerate(dataset)):
        frames = batch[0].to(device)
        operations = batch[1].numpy()

        model.rule_network.reset()
        outputs = model(frames)
        loss = metric(outputs.flatten(0, 1), frames[:, 1:, :, :].flatten(0, 1))
        losses.append(loss.item())

        rule_golden.append(operations[:, :, -1])
        rule_actual.append(np.array(model.rule_network.rule_selection))
        var_golden.append(operations[:, :, 0])
        var_actual.append(np.array(model.rule_network.primary_selection))

    val_loss = np.mean(losses)

    rule_golden = rearrange(rule_golden, 'n b t-> (n b t)')
    rule_actual = rearrange(rule_actual, 'n t b -> (n b t)')
    var_golden = rearrange(var_golden, 'n b t -> (n b t)')
    var_actual = rearrange(var_actual, 'n t b -> (n b t)')

    map_rule_ga = count_mapping(rule_golden, rule_actual, range(n_rule))
    map_var_ga = count_mapping(var_golden, var_actual, range(n_slot))

    return val_loss, map_rule_ga, map_var_ga


def main():
    n_epoch = 300
    batch_size = 64
    lr0 = 0.0001
    n_op = 1
    n_rule = 4
    c_rule = 12
    n_slot = 2
    c_slot = 6
    device = pt.device('cuda')

    dload_t = ptud.DataLoader(
        CoordinateArithmeticSet(n_slot=n_slot, n_operation=n_op, n_example=10000),
        batch_size, True, num_workers=4, collate_fn=None, drop_last=True
    )
    dloat_v = ptud.DataLoader(
        CoordinateArithmeticSet(n_slot=n_slot, n_operation=n_op, n_example=2000),
        batch_size, False, num_workers=4, collate_fn=None, drop_last=True
    )
    model = CoordinateArithmeticNps(n_rule, c_rule, n_slot, c_slot).to(device)
    metric = ptn.MSELoss()
    optim = pto.Adam(model.parameters(), lr=lr0)

    ####
    # frames = dload_t.dataset.__getitem__(0)[0][None].to(device)
    # outputs = model(frames)
    # model.rule_network.reset()
    ####

    ckpt_path = './res/'
    best_ckpt_file = osp.join(ckpt_path, f'best.pth')
    if not osp.exists(ckpt_path):
        os.makedirs(ckpt_path)

    loss_min = 10.
    for n in range(n_epoch):
        loss_v, map_op_rule_v, map_num_slot_v = epoch_v(model, dloat_v, metric, device, n_slot, n_rule)
        print(colored(
            f'[val epoch{n:04d}]\tloss={loss_v:.10f}\n{map_op_rule_v}\n{map_num_slot_v}',
            'cyan'))
        loss_t, map_op_rule_t, map_num_slot_t = epoch_t(model, dload_t, metric, optim, device, n_slot, n_rule)
        print(
            f'[train epoch{n:04d}]\tloss={loss_t:.10f}\n{map_op_rule_t}\n{map_num_slot_t}'
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
    set_seed(6)
    main()

# TODO 减少神经规则容量，可能更有助于专网专用
