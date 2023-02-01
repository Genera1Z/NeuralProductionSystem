import os
import os.path as osp
import sys
import time

import numpy as np
import torch as pt
import torch.backends.cudnn as ptbc
import torch.nn as ptn
import torch.optim as pto
import torch.utils.data as ptud
from tqdm import tqdm

from datum import ArithmeticSet
from learn import NormalizedMse
from model_official_strict import ArithmeticNps

# from model_official_shareqkattent import ArithmeticNps
# from model_official_bargmax_bselect import ArithmeticNps
# from model_official import ArithmeticNps
# from model_strict import ArithmeticNps


METRICS = {'NormalizedMse': NormalizedMse, 'MSE': ptn.MSELoss}


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
    operators = operators.astype('int')
    selections = selections.astype('int')

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


def epoch_t(
        model, dataset, metric, optim, tau
):
    model.train()
    loss_ttl = []
    op_ttl = []
    sel_ttl = []

    for i, batch in tqdm(enumerate(dataset)):
        ____ = dataset.dataset.organize(batch.cuda())
        inps_prev, inps_curr, ops_curr, tars_curr = [_.view(-1) for _ in ____]

        model.reset()  # TODO 改成串行版本，只用最后的loss

        model.tau1 = tau
        model.tau2 = tau

        oups = model(inps_prev, inps_curr, ops_curr)
        loss = metric(oups, tars_curr)

        model.zero_grad()
        loss.backward()
        optim.step()

        loss_ttl.append(loss.detach().cpu().numpy())
        op_ttl.append(ops_curr.cpu())
        sel_ttl.extend(model.rule_selection)

    loss_avg = np.mean(loss_ttl)  # ()
    operations = np.concatenate(op_ttl, axis=0)  # (n,)
    selections = np.concatenate(sel_ttl, axis=0)  # (n,)
    op_to_rule, rule_to_op = count_rule_selections(operations, selections)

    return loss_avg, op_to_rule, rule_to_op


def epoch_v(
        model, dataset, metric
):
    model.eval()
    loss_ttl = []
    op_ttl = []
    sel_ttl = []

    nograd = pt.no_grad()
    nograd.__enter__()

    length = 1
    for i, batch in tqdm(enumerate(dataset)):
        ____ = dataset.dataset.organize(batch.cuda())
        length = batch.size(1)

        for t in range(length):
            inp_prev_t, inp_curr_t, op_curr_t, tar_curr_t = [_[:, t] for _ in ____]
            model.reset()
            oup_t = model(inp_prev_t, inp_curr_t, op_curr_t)
            loss = metric(oup_t, tar_curr_t)  # TODO 只算最后一步的？

            loss_ttl.append(loss.cpu().numpy())
            op_ttl.append(op_curr_t.cpu())
            sel_ttl.extend(model.rule_selection)

    nograd.__exit__(None, None, None)

    loss_avg = np.mean(loss_ttl) * length  # ()  # XXX 所有步的累积损失
    operations = np.concatenate(op_ttl, axis=0)  # (n,)
    selections = np.concatenate(sel_ttl, axis=0)  # (n,)
    op_to_rule, rule_to_op = count_rule_selections(operations, selections)

    return loss_avg, op_to_rule, rule_to_op


def main_t(
        n_xmpl_t=10240, n_xmpl_v=2560, length=20, batch_size=64, n_epoch=100,
):
    xmpls_t = ptud.DataLoader(ArithmeticSet(length, n_xmpl_t), batch_size, True)
    xmpls_v = ptud.DataLoader(ArithmeticSet(length, n_xmpl_v), batch_size, False)

    model = ArithmeticNps(64, 3, 32, share=[bool(int(_)) for _ in share])
    model.cuda()
    metric = METRICS[goal]()
    optim = pto.Adam(model.parameters(), lr=1e-4)

    ckpt_path = './res/'
    best_ckpt_file = osp.join(ckpt_path, f'best-{seed}-{share}-{goal}.pth')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    loss_min = 10.

    for e in range(n_epoch):
        tau = n_epoch / (e + 1)
        # print(tau)
        loss_t, op_to_rule_t, rule_to_op_t = epoch_t(model, xmpls_t, metric, optim, tau)
        print(f'[train epoch{e:04d}]\tloss={loss_t:.10f}\n{op_to_rule_t}\n{rule_to_op_t}')
        loss_v, op_to_rule_v, rule_to_op_v = epoch_v(model, xmpls_v, metric)
        print(f'[val epoch{e:04d}]\tloss(*length)={loss_v:.10f}\n{op_to_rule_v}\n{rule_to_op_v}')
        if loss_v < loss_min:
            loss_min = loss_v
            print('...saving new best model checkpoint...')
            pt.save(model, best_ckpt_file)


def main_e(
        n_xmpl_e=2560, lengths=(10, 20, 30, 40, 50), batch_size=64
):
    ckpt_path = './res/'
    best_ckpt_file = osp.join(ckpt_path, f'best-{seed}-{share}-{goal}.pth')
    model = pt.load(best_ckpt_file)
    metric = METRICS[goal]()
    losses = []
    for l in lengths:
        xmpls_e = ptud.DataLoader(ArithmeticSet(l, n_xmpl_e), batch_size, False)
        loss_e, op_to_rule_e, rule_to_op_e = epoch_v(model, xmpls_e, metric)
        print(f'[test length{l:04d}]\tloss(*length)={loss_e:.10f}\n{op_to_rule_e}\n{rule_to_op_e}')
        losses.append(loss_e)
    loss_save_file = osp.join(ckpt_path, f'loss-{seed}-{share}-{goal}.txt')
    np.savetxt(loss_save_file, [losses], fmt='%.8e', delimiter=',')


def set_seed(
        # seed=20221230
):
    print(f'seed={seed}')
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    ptbc.deterministic = True
    pt.manual_seed(seed)
    pt.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    print('####', sys.argv, '####')
    t0 = time.time()
    seed = int(sys.argv[1])
    share = str(sys.argv[2])
    goal = str(sys.argv[3])
    assert len(share) == 4
    assert goal in METRICS.keys()
    set_seed()
    main_t()
    main_e()
    print('time=', time.time() - t0)
