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
            cv, nv, nr=4, cr=64
    ):
        super().__init__()
        self.device = pt.device('cuda') if pt.cuda.is_available() else pt.device('cpu')

        w = pt.randn(1, nr, cr).to(self.device)
        self.rule_embeddings = nn.Parameter(w)
        self.rule_mlp = nn.Sequential(GLinear(4, 32, nr), nn.Dropout(0.1), nn.ReLU(), GLinear(32, 2, nr))
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
        rule_emb = self.rule_embeddings.repeat(b, 1, 1)

        q1 = rule_emb
        k1 = hidden
        attent1 = self.dropout(self.selector1(q1, k1))
        shape1 = attent1.shape
        attent1_ = attent1.flatten(1)
        if self.training:
            mask1_ = ptnf.gumbel_softmax(attent1_, tau=1.0, hard=True, dim=1)
        else:
            mask1_ = argmax_onehot(attent1_, dim=1)
        mask1 = mask1_.view(*shape1)  # (b,nr,nv)

        mask_r = mask1.sum(dim=2).unsqueeze(-1)
        mask_p = mask1.sum(dim=1).unsqueeze(-1)
        var_p = (hidden * mask_p).sum(dim=1)

        self.rule_selection.append(pt.argmax(mask_r.detach()[:, :, 0], dim=1).cpu().numpy())
        self.primary_selection.append(pt.argmax(mask_p.detach()[:, :, 0], dim=1).cpu().numpy())

        q2 = var_p.unsqueeze(1)
        k2 = hidden
        attent2 = self.dropout(self.selector2(q2, k2)).squeeze(1)
        if self.training:
            mask2 = ptnf.gumbel_softmax(attent2, tau=1.0, hard=True, dim=1)
        else:
            mask2 = argmax_onehot(attent2, dim=1)

        var_c = (hidden * mask2.unsqueeze(-1)).sum(dim=1)

        self.context_selection.append(pt.argmax(mask2.detach(), dim=1).cpu().numpy())

        h_dim = var_p.size(1)  # TODO XXX ??? 为啥要截取通道的前三分之一？
        var_c = var_c[:, :h_dim // 3].unsqueeze(1).repeat(1, rule_emb.size(1), 1)
        var_p = var_p[:, :h_dim // 3].unsqueeze(1).repeat(1, rule_emb.size(1), 1)

        var_pc = pt.cat((var_p, var_c), dim=-1)  # TODO shape=?
        # TODO XXX 此处若多乘一次mask_r，前向等价，但后向恶化！
        output = (self.rule_mlp(var_pc) * mask_r).sum(dim=1).unsqueeze(1)

        output = output.repeat(1, hidden.size(1), 1)
        output = output * mask_p  # TODO how about remove this?

        return output, mask_p

    def reset(self):
        self.rule_selection.clear()
        self.primary_selection.clear()
        self.context_selection.clear()


class CoordinateArithmeticNps(nn.Module):

    def __init__(self,
            num_slots, num_rules, rule_dim=3, dropout=0.1, rule_mlp_dim=32
    ):
        super().__init__()
        self.rule_network = RuleNetwork(6, num_slots, nr=num_rules, cr=rule_dim)
        self.mse_loss = nn.MSELoss()  # TODO move out to main

    def forward(self, x):
        b, t, _, _ = x.size()
        aux_input = x[:, -1]
        state = x[:, 0]

        mse_loss = 0

        predicted_number = 0
        for i in range(t - 1):
            diff = aux_input - state
            state = pt.cat((state, diff, aux_input), dim=-1)  # (64,2,6)

            state_, mask = self.rule_network(state)
            state = mask * state_ + (1 - mask) * state[:, :, :2]

            loss_ = self.mse_loss(state, aux_input)
            mse_loss += loss_

            predicted_number = state

        return mse_loss, predicted_number, self.rule_network.rule_selection, self.rule_network.primary_selection
