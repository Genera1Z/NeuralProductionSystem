import numpy as np
import torch as pt
import torch.utils.data as ptud


class SequentialArithmeticSet(ptud.Dataset):
    OP_SET = (np.add, np.subtract, np.multiply)

    def __init__(self, length, n_xmpl):
        self.length = length
        self.n_xmpl = n_xmpl
        self.inps, self.tars, self.ops = self._generate_example()

    def _generate_example(self):
        inps = []
        tars = []
        ops = []
        tar = np.zeros([self.n_xmpl])
        for i in range(self.length):
            inp = np.round(np.random.uniform(0, 1, self.n_xmpl), decimals=2)  # TODO 0~2
            op = np.random.choice([0, 1, 2], self.n_xmpl)
            for _ in [0, 1, 2]:
                idxs = np.where(op == _)[0]
                tar[idxs] = self.OP_SET[_](tar[idxs], inp[idxs])
            inps.append(inp)
            tars.append(tar.copy())
            ops.append(op)
        inps = np.stack(inps, axis=1).astype('float32')
        tars = np.stack(tars, axis=1).astype('float32')
        ops = np.stack(ops, axis=1).astype('float32')
        return inps, tars, ops

    @staticmethod
    def organize(batch):
        assert len(batch.shape) == 3
        inps, tars, ops = [batch[:, :, _] for _ in range(3)]
        inps_prev = pt.cat([pt.zeros_like(tars[:, :1]), tars[:, :-1]], dim=1)  # .view(-1)
        inps_curr = inps  # .view(-1)
        ops_curr = ops  # .view(-1)
        tars_curr = tars  # .view(-1)
        return inps_prev, inps_curr, ops_curr, tars_curr

    def __getitem__(self, item):
        return np.stack([self.inps[item], self.tars[item], self.ops[item]], axis=1)

    def __len__(self):
        return self.n_xmpl


########################################################################################################################


def main():
    xmpls = SequentialArithmeticSet(20, 25600)
    for _ in xmpls:
        inp, tar, op = [_[:, i] for i in range(3)]
        tar_ = np.concatenate([np.zeros([1]), tar[:-1]])
        for i in range(3):
            idx = np.where(op == i)[0]
            assert np.all(tar[idx] == xmpls.OP_SET[i](tar_[idx], inp[idx]))

    return


if __name__ == '__main__':
    main()
