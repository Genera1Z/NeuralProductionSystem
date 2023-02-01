import os
import os.path as osp

import numpy as np
import torch as pt
import torch.nn as ptn
import torch.optim as pto
import torch.utils.data as ptud
from tqdm import tqdm

from datum import ArithmeticSet
# from model_official import ArithmeticNps
from model_strict import ArithmeticNps


def count_rule_selections(
        operators: np.ndarray, selections: np.ndarray, string=True
):
    """
    :param operators: in shape (n,)
    :param selections: in shape (n,)
    :param string: bool
    :return:
    """
    assert len(operators.shape) == len(selections.shape) == 1
    assert len(operators) == len(selections)
    assert operators.dtype == selections.dtype == np.int

    func = lambda _, a, b: dict(zip(*np.unique(b[np.where(a == _)[0]], return_counts=True)))

    cnts_r = [func(_, operators, selections) for _ in range(3)]  # [{0:a,1:b,2:c},..]
    op_to_rule = {k: v for k, v in zip('+-*', cnts_r)}

    cnts_o = [func(_, selections, operators) for _ in range(3)]  # [{0:x,1:y,2:z},..]
    for cnt_o in cnts_o:
        for idx_r, idx_o in zip([0, 1, 2], '+-*'):
            if idx_r in cnt_o:
                cnt_o[idx_o] = cnt_o.pop(idx_r)
    rule_to_op = {k: v for k, v in zip('012', cnts_o)}

    if string:
        op_to_rule, rule_to_op = [str(_).replace("'", '') for _ in (op_to_rule, rule_to_op)]
    return op_to_rule, rule_to_op


def execute_epoch(
        model, dataset, metric, optim=None
):
    if optim is not None:
        model.train()
    else:
        model.eval()
    loss_ttl = []
    op_ttl = []
    sel_ttl = []

    if optim is None:
        nograd = pt.no_grad()
        nograd.__enter__()
    else:
        nograd = ...

    for i, batch in tqdm(enumerate(dataset)):
        ____ = dataset.dataset.organize(batch.cuda())
        inps_prev, inps_curr, ops_curr, tars_curr = [_.view(-1) for _ in ____]

        model.reset()
        loss = 0.
        for t in range(batch.size(1)):
            inp_prev_t, inp_curr_t, op_curr_t, tar_curr_t = [_[:, t] for _ in ____]
            oup_t = model(inp_prev_t, inp_curr_t, op_curr_t)
            loss += metric(oup_t, tar_curr_t)

        loss_ttl.append(loss.detach().cpu())
        op_ttl.append(ops_curr.cpu())
        sel_ttl.extend(model.rule_selection)

        if optim is not None:
            model.zero_grad()
            loss.backward()
            optim.step()
        else:
            ...

    if optim is not None:
        ...
    else:
        nograd.__exit__(None, None, None)

    loss_avg = np.mean(loss_ttl)  # ()
    operations = np.concatenate(op_ttl, axis=0).astype('int')  # (n,)
    selections = np.concatenate(sel_ttl, axis=0).astype('int')  # (n,)
    op_to_rule, rule_to_op = count_rule_selections(operations, selections)

    return loss_avg, op_to_rule, rule_to_op


def main_train(
        n_xmpl_t=10240, n_xmpl_v=2560, length=20, batch_size=64, n_epoch=100,
):
    xmpls_t = ptud.DataLoader(ArithmeticSet(length, n_xmpl_t), batch_size, True)
    xmpls_v = ptud.DataLoader(ArithmeticSet(length, n_xmpl_v), batch_size, False)

    model = ArithmeticNps(64, 3, 32)
    model.cuda()
    metric = ptn.MSELoss()
    optim = pto.Adam(model.parameters(), lr=1e-4)

    ckpt_path = './res/'
    best_ckpt_file = osp.join(ckpt_path, f'best-{model.__class__.__name__}.pth')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    loss_min = 1.

    for e in range(n_epoch):
        loss_t, op_to_rule_t, rule_to_op_t = execute_epoch(model, xmpls_t, metric, optim)
        print(f'[epoch{e:04d}] train\nloss={loss_t:.10f}\n{op_to_rule_t}\n{rule_to_op_t}')
        loss_v, op_to_rule_v, rule_to_op_v = execute_epoch(model, xmpls_v, metric)
        print(f'[epoch{e:04d}] val\nloss(*length)={loss_v:.10f}\n{op_to_rule_v}\n{rule_to_op_v}')
        if loss_v < loss_min:
            print('saving new best model checkpoint...')
            pt.save(model, best_ckpt_file)


def main_eval(
        n_xmpl_e=2560, lengths=(10, 20, 30, 40, 50), batch_size=64
):
    ckpt_path = './res/'
    best_ckpt_file = osp.join(ckpt_path, f'best-{ArithmeticNps.__name__}.pth')
    model = pt.load(best_ckpt_file)
    metric = ptn.MSELoss()
    for l in lengths:
        xmpls_e = ptud.DataLoader(ArithmeticSet(l, n_xmpl_e), batch_size, False)
        loss, op_to_rule, rule_to_op = execute_epoch(model, xmpls_e, metric)
        print(f'[length{l:04d}] test\nloss(*length)={loss:.10f}\n{op_to_rule}\n{rule_to_op}')


if __name__ == '__main__':
    main_train()
    main_eval()
