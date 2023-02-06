import math

import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as ptnf


def argmax_onehot(x: pt.Tensor, dim: int):
    idx = x.argmax(dim=dim)
    onehot = pt.zeros_like(x).scatter_(dim, idx.unsqueeze(dim), 1.0)
    return onehot


class GLinear(nn.Module):

    def __init__(self, din, dout, num_blocks, bias=True, a=None):
        super(GLinear, self).__init__()
        if a is None:
            a = 1. / math.sqrt(dout)
        self.weight = nn.Parameter(pt.FloatTensor(num_blocks, din, dout).uniform_(-a, a))
        self.bias = bias
        if bias is True:
            self.bias = nn.Parameter(pt.FloatTensor(num_blocks, dout).uniform_(-a, a))
        else:
            self.bias = None

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = pt.bmm(x, self.weight)
        x = x.permute(1, 0, 2)
        if not self.bias is None:
            x = x + self.bias
        return x


class SelectAttention(nn.Module):

    def __init__(self, cq, ck, cm=16, nq=5, nk=5, share_q=False, share_k=False):
        super(SelectAttention, self).__init__()
        self.proj_q = nn.Linear(cq, cm) if share_q else GLinear(cq, cm, nq)
        self.proj_k = nn.Linear(ck, cm) if share_k else GLinear(ck, cm, nk)
        self.temperature = np.sqrt(cm)

    def forward(self, q, k):
        r = self.proj_q(q)
        w = self.proj_k(k)
        a = pt.bmm(r, w.permute(0, 2, 1)) / self.temperature
        return a


class RuleNetwork(nn.Module):

    def __init__(self,
            nr, cr, nv, cv
    ):
        super().__init__()
        self.rule_embeds = nn.Parameter(pt.randn(nr, cr))
        self.rule_mlps = nn.Sequential(GLinear(4, 32, nr), nn.Dropout(0.1), nn.ReLU(), GLinear(32, 2, nr))
        # GMlpD2(4, 2, nr, do=0.1, cm=32)  # TODO 为啥是4、2？

        self.dropout = nn.Dropout(p=0.1)

        self.selector1 = SelectAttention(cr, cv, cm=32, nq=nr, nk=nv, share_q=True, share_k=True)
        self.selector2 = SelectAttention(cv, cv, cm=16, nq=1, nk=nv, share_q=True, share_k=False)

        print(self)

        self.rule_selection = []
        self.primary_selection = []
        self.context_selection = []

    def forward(self, hidden):
        b, nv, cv = hidden.size()
        nr, cr = self.rule_embeds.size()
        rule_embeds = self.rule_embeds[None, :, :].repeat(b, 1, 1)

        # 1 # select rule and primary

        q1 = rule_embeds
        k1 = hidden
        attent1 = self.dropout(self.selector1(q1, k1))
        shape1 = attent1.shape
        attent1_ = attent1.flatten(1)
        if self.training:
            mask1_ = ptnf.gumbel_softmax(attent1_, tau=1.0, hard=True, dim=1)
        else:
            mask1_ = argmax_onehot(attent1_, dim=1)
        mask1 = mask1_.view(*shape1)  # (b,nr,nv)

        mask_r = mask1.sum(dim=2)
        mask_p = mask1.sum(dim=1)
        var_p = (hidden * mask_p[:, :, None]).sum(dim=1)

        self.rule_selection.append(pt.argmax(mask_r.detach(), dim=1).cpu().numpy())
        self.primary_selection.append(pt.argmax(mask_p.detach(), dim=1).cpu().numpy())

        # 2 # select context

        q2 = var_p[:, None, :]
        k2 = hidden
        attent2 = self.dropout(self.selector2(q2, k2))[:, 0, :]
        if self.training:
            mask2 = ptnf.gumbel_softmax(attent2, tau=1.0, hard=True, dim=1)
        else:
            mask2 = argmax_onehot(attent2, dim=1)

        var_c = (hidden * mask2[:, :, None]).sum(dim=1)

        self.context_selection.append(pt.argmax(mask2.detach(), dim=1).cpu().numpy())

        # 3 # apply rule and update primary

        var_c2 = var_c[:, None, :nv].repeat(1, nr, 1)
        var_p2 = var_p[:, None, :nv].repeat(1, nr, 1)

        var_pc = pt.cat([var_p2, var_c2], dim=2)
        output = (self.rule_mlps(var_pc) * mask_r[:, :, None]).sum(dim=1)
        # XXX 此处若多乘一次mask_r，前向等价，但后向恶化！

        output = output[:, None, :].repeat(1, nv, 1)
        output = output * mask_p[:, :, None]  # TODO 好像跟外边的mask乘法重复了
        # TODO 应该改成直接更新hidden的方式，输入hidden输出hidden，三个NPS都应该这样。参考那个大佬的NPS实现。

        return output, mask_p

    def reset(self):
        self.rule_selection.clear()
        self.primary_selection.clear()
        self.context_selection.clear()


class CoordinateArithmeticNps(nn.Module):

    def __init__(self,
            nr, cr, nv, cv
    ):
        super().__init__()
        self.rule_network = RuleNetwork(nr, cr, nv, cv)

    def forward(self, inputs):
        outputs = []
        frame = inputs[:, 0, :, :]

        for t in range(1, inputs.size(1)):
            target = inputs[:, t, :, :]
            diff = target - frame
            frame_ = pt.cat([frame, diff, target], dim=2)

            frame2, mask = self.rule_network(frame_)

            frame = mask[:, :, None] * frame2 + (1 - mask[:, :, None]) * frame
            # frame = frame2 + (1 - mask[:, :, None]) * frame  # TODO 更差？？？去掉随机种子，多试几次？
            outputs.append(frame)

        return pt.stack(outputs, dim=1)
