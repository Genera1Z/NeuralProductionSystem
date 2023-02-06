import os
import os.path as osp
import time

import numpy as np
import torch as pt
import torch.backends.cudnn as ptbc
import torch.nn as ptn
import torch.optim as pto
import torch.utils.data as ptud
from einops import rearrange
from termcolor import colored
from tqdm import tqdm

from datum import SequentialArithmeticSet
from learn import NormalizedMse
from model_official_shareqkattent import SequentialArithmeticNps  # XXX


# from model_official_strict import SequentialArithmeticNps  # XXX
# from model_official_bargmax_bselect import ArithmeticNps
# from model_official import ArithmeticNps
# from model_strict import ArithmeticNps


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


def epoch_t(model, dataset, metric, optim, device, n_slot, n_rule, tau):
    model.train()

    losses = []
    rule_golden = []
    rule_actual = []
    var_actual = []

    for i, batch in tqdm(enumerate(dataset)):
        ____ = dataset.dataset.organize(batch.to(device))
        inps_prev, inps_curr, ops_curr, tars_curr = [_.view(-1) for _ in ____]

        model.reset()  # TODO 改成串行版本，只用最后的loss
        optim.zero_grad()

        # model.tau1 = tau
        # model.tau2 = tau

        oups = model(inps_prev, inps_curr, ops_curr)
        loss = metric(oups, tars_curr)
        losses.append(loss.item())

        loss.backward()
        optim.step()

        rule_golden.append(ops_curr.cpu().numpy())  # (b*t,)
        rule_actual.append(np.array(model.rule_selection))  # (1,b*t)
        var_actual.append(np.array(model.primary_selection))  # (1,b*t)

    train_loss = np.mean(losses)

    # rule_golden = rearrange(rule_golden, 'n b t -> (n b t)')
    # rule_actual = rearrange(rule_actual, 'n t b -> (n b t)')
    # var_actual = rearrange(var_actual, 'n t b -> (n b t')
    rule_golden = rearrange(rule_golden, 'n bt -> (n bt)')
    rule_actual = rearrange(rule_actual, 'n 1 b -> (n b 1)')
    var_actual = rearrange(var_actual, 'n 1 b -> (n b 1)')

    map_rule_ga = count_mapping(rule_golden, rule_actual, range(n_rule))
    map_rule_var = count_mapping(rule_actual, var_actual, range(n_slot))

    return train_loss, map_rule_ga, map_rule_var


@pt.no_grad()
def epoch_v(model, dataset, metric, device, n_slot, n_rule):
    model.eval()

    losses = []
    rule_golden = []
    rule_actual = []
    var_actual = []

    for i, batch in tqdm(enumerate(dataset)):
        ____ = dataset.dataset.organize(batch.to(device))

        model.reset()

        outputs = []
        for t in range(batch.size(1)):
            inp_prev_t, inp_curr_t, op_curr_t, tar_curr_t = [_[:, t] for _ in ____]
            oup_t = model(inp_prev_t, inp_curr_t, op_curr_t)
            outputs.append(oup_t)
        outputs = pt.stack(outputs, dim=1)

        loss = metric(outputs.flatten(0, 1), ____[3].flatten(0, 1))  # TODO 只算最后一步
        losses.append(loss.item())

        rule_golden.append(____[2].cpu().numpy())
        rule_actual.append(np.array(model.rule_selection))
        var_actual.append(np.array(model.primary_selection))

    var_loss = np.mean(losses)

    rule_golden = rearrange(rule_golden, 'n b t -> (n b t)')
    rule_actual = rearrange(rule_actual, 'n t b -> (n b t)')
    var_actual = rearrange(var_actual, 'n t b -> (n b t)')

    map_rule_ga = count_mapping(rule_golden, rule_actual, range(n_rule))
    map_rule_var = count_mapping(rule_actual, var_actual, range(n_slot))

    return var_loss, map_rule_ga, map_rule_var


METRICS = {'NormalizedMse': NormalizedMse, 'MSE': ptn.MSELoss}


def main():
    n_epoch = 100
    batch_size = 64
    lr0 = 0.0001
    n_op = 20
    n_rule = 3
    c_rule = 32
    n_slot = 3
    c_slot = 64
    device = pt.device('cuda')

    dload_t = ptud.DataLoader(
        SequentialArithmeticSet(n_op, 10240),
        batch_size, True, num_workers=1, collate_fn=None, drop_last=True
    )
    dload_v = ptud.DataLoader(
        SequentialArithmeticSet(n_op, 2560),
        batch_size, True, num_workers=1, collate_fn=None, drop_last=True
    )
    model = SequentialArithmeticNps(n_rule, c_rule, n_slot, c_slot).to(device)  # 64, 3, 32,
    metric = METRICS['MSE']()
    optim = pto.Adam(model.parameters(), lr=lr0)

    ckpt_path = './res/'
    # best_ckpt_file = osp.join(ckpt_path, f'best-{seed}-{share}-{goal}.pth')
    best_ckpt_file = osp.join(ckpt_path, f'best.pth')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    loss_min = 10.
    for e in range(n_epoch):
        tau = n_epoch / (e + 1)
        # print(tau)
        loss_v, map_rule_ga_v, map_rule_var_v = epoch_v(model, dload_v, metric, device, n_slot, n_rule)
        print(colored(
            f'[val epoch{e:04d}]\tloss(*length)={loss_v:.10f}\n{map_rule_ga_v}\n{map_rule_var_v}',
            'green'))
        loss_t, map_rule_ga_t, map_rule_var_t = epoch_t(model, dload_t, metric, optim, device, n_slot, n_rule, tau)
        print(
            f'[train epoch{e:04d}]\tloss={loss_t:.10f}\n{map_rule_ga_t}\n{map_rule_var_t}'
        )
        if loss_v < loss_min:
            loss_min = loss_v
            print('...saving new best model checkpoint...')
            pt.save(model, best_ckpt_file)

    for l in [10, 20, 30, 40, 50]:
        dload_e = ptud.DataLoader(
            SequentialArithmeticSet(l, 2560),
            batch_size, False, num_workers=1, collate_fn=None, drop_last=True
        )
        model_e = pt.load(best_ckpt_file)
        loss_e, map_rule_ga_e, map_rule_var_e = epoch_v(model_e, dload_e, metric, device, n_slot, n_rule)
        print(
            f'[eval length{l:04d}]\tloss={loss_e:.10f}\n{map_rule_ga_e}\n{map_rule_var_e}'
        )


def set_seed(
        seed=20221230
):
    print(f'seed={seed}')
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    ptbc.deterministic = True
    pt.manual_seed(seed)
    pt.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    # print('####', sys.argv, '####')
    t0 = time.time()
    # seed = int(sys.argv[1])
    # share = str(sys.argv[2])
    # goal = str(sys.argv[3])
    # assert len(share) == 4
    # assert goal in METRICS.keys()
    # set_seed(3)
    main()
    print('time=', time.time() - t0)
