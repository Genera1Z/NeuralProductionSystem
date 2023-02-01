import os
import os.path as osp
import random

import numpy as np
import torch as pt
import torch.backends.cudnn as ptbc
import torch.optim as pto
import torch.utils.data as ptud
from einops import rearrange
from termcolor import colored
from tqdm import tqdm

from datum import CoordinateArithmeticSet
from model import CoordinateArithmeticNps


def rule_stats(rule_activations, variable_activation, operations, number_to_slot, operation_to_rule):
    for b in range(rule_activations[0].shape[0]):
        for t in range(len(rule_activations)):
            var_selected = variable_activation[t][b]
            number_key = operations[b][t][0]
            number_to_slot[number_key][var_selected] += 1

            rule_selected = rule_activations[t][b]
            operation_key = operations[b][t][-1]
            operation_to_rule[operation_key][rule_selected] += 1

    return number_to_slot, operation_to_rule


def count_mapping(s: np.ndarray, t: np.ndarray, keys, return_string=True):
    assert len(s.shape) == len(t.shape) == 1
    assert len(s) == len(t)
    # assert s.dtype in (np.uint8, np.uint16, np.uint32, np.uint64)
    # assert t.dtype in (np.uint8, np.uint16, np.uint32, np.uint64)
    func = lambda _, a, b: dict(zip(*np.unique(b[np.where(a == _)[0]], return_counts=True)))
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
        x = batch[0].float().to(device)  # (b,t,c,d)
        ops = batch[1].numpy()  # (b,t,3)

        model.rule_network.reset()
        optim.zero_grad()
        loss, preds, rule_activations, variable_activation = model(x)

        optim.zero_grad()
        loss.backward()
        optim.step()

        losses.append(loss.item())

        rule_golden.append(ops[:, :, -1])
        rule_actual.append(np.array(rule_activations))
        var_golden.append(ops[:, :, 0])
        var_actual.append(np.array(variable_activation))

    train_loss = np.mean(losses)

    rule_golden = rearrange(rule_golden, 'n b t-> (n b t)')
    rule_actual = rearrange(rule_actual, 'n t b -> (n b t)')
    var_golden = rearrange(var_golden, 'n b t -> (n b t)')
    var_actual = rearrange(var_actual, 'n t b -> (n b t)')

    map_rule_ga = count_mapping(rule_golden, rule_actual, range(n_rule))
    map_var_ga = count_mapping(var_golden, var_actual, range(n_slot))

    return train_loss, map_rule_ga, map_var_ga


def epoch_v(model, dataset, metric, device, n_slot, n_rule):
    model.eval()

    losses = []
    rule_golden = []
    rule_actual = []
    var_golden = []
    var_actual = []

    for i, batch in tqdm(enumerate(dataset)):
        x = batch[0].float().to(device)
        ops = batch[1].numpy()

        model.rule_network.reset()
        loss, preds, rule_activations, variable_activation = model(x)

        losses.append(loss.item())

        rule_golden.append(ops[:, :, -1])
        rule_actual.append(np.array(rule_activations))
        var_golden.append(ops[:, :, 0])
        var_actual.append(np.array(variable_activation))

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
    n_slot = 2
    n_rule = 4
    c_rule = 12
    device = pt.device('cuda')

    train_dataset = CoordinateArithmeticSet(n_operation=1, n_slot=n_slot, n_example=10000)
    eval_dataset = CoordinateArithmeticSet(n_operation=1, n_slot=n_slot, n_example=2000)
    dload_t = ptud.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=None,
        drop_last=True)
    dloat_v = ptud.DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=None,
        drop_last=True)

    model = CoordinateArithmeticNps(n_slot, n_rule, c_rule, 0.35, 16).cuda()
    optim = pto.Adam(model.parameters(), lr=lr0)

    ckpt_path = './res/'
    best_ckpt_file = osp.join(ckpt_path, f'best-.pth')
    if not osp.exists(ckpt_path):
        os.makedirs(ckpt_path)

    ####
    # x = dload_t.dataset.__getitem__(0)[0][None, ...].repeat(64, 1, 1, 1).float()
    # assert tuple(x.shape) == (64, 2, 2, 2)
    # model.train()
    # loss_, preds, rule_activations, variable_activation = model(x)
    ####

    loss_min = 10
    for n in range(n_epoch):
        loss_v, map_op_rule_v, map_num_slot_v = epoch_v(model, dloat_v, None, device, n_slot, n_rule)
        print(colored(f'[val epoch{n:04d}]\tloss={loss_v:.10f}\n{map_op_rule_v}\n{map_num_slot_v}', 'cyan'))
        loss_t, map_op_rule_t, map_num_slot_t = epoch_t(model, dload_t, None, optim, device, n_slot, n_rule)
        print(f'[train epoch{n:04d}]\tloss={loss_t:.10f}\n{map_op_rule_t}\n{map_num_slot_t}')

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
