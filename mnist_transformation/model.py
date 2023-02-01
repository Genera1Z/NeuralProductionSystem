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
        self.rule_embeddings = nn.Parameter(w)  # XXX better init? orth seems not useful
        self.rule_mlp = nn.Sequential(GLinear(cv, 128, nr), nn.Dropout(0.1), nn.ReLU(), GLinear(128, cv, nr))

        self.dropout = nn.Dropout(p=0.1)  # XXX seems effective

        self.selector1 = SelectAttention(cr, cv, cm=32, nq=nr, nk=nv, share_q=True, share_k=True)
        self.selector2 = SelectAttention(cr, cv, cm=16, nq=1, nk=2, share_q=True, share_k=True)

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
        var_r = (rule_emb * mask_r).sum(dim=1)

        self.rule_selection.append(pt.argmax(mask_r.detach()[:, :, 0], dim=1).cpu().numpy())

        q2 = var_r.unsqueeze(1)
        k2 = hidden
        attent2 = self.dropout(self.selector2(q2, k2)).squeeze(1)
        if self.training:
            mask2 = ptnf.gumbel_softmax(attent2, tau=0.5, hard=True, dim=1)
        else:
            mask2 = argmax_onehot(attent2, dim=1)

        var_p = (hidden * mask2.unsqueeze(-1)).sum(dim=1)

        self.primary_selection.append(pt.argmax(mask2.detach(), dim=1).cpu().numpy())

        input = var_p.unsqueeze(1).repeat(1, rule_emb.size(1), 1)
        output = self.rule_mlp(input)
        output = (output * mask_r).sum(dim=1).unsqueeze(1)

        return output

    def reset(self):
        self.rule_selection.clear()
        self.primary_selection.clear()
        self.context_selection.clear()


class Reshape(nn.Module):

    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(input.size(0), *self.shape)


class LayerNorm(nn.Module):

    def forward(self, input):
        return ptnf.layer_norm(input, input.size()[1:])


class MnistOperationNps(nn.Module):
    """Encoder from https://openreview.net/pdf?id=ryH20GbRW."""

    def __init__(self, nr, cr, nv, cv):
        super(MnistOperationNps, self).__init__()
        self.encoder = self.build_encoder(1, 64, 64, cv)
        self.operation_rep = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, cv))
        self.rule_network = RuleNetwork(cv, nv, nr=nr, cr=cr)
        self.decoder = self.build_decoder(cv, 8, 8, 1)

    def forward(self, frames, operations):
        b, t = frames.shape[:2]

        encoded_f = self.encoder(frames.flatten(0, 1)).unsqueeze(1)
        encoded_o = self.operation_rep(operations.flatten(0, 1)).unsqueeze(1)
        hidden = pt.cat((encoded_f, encoded_o), dim=1)  # (b,nv,cv)
        hidden = hidden.view(b, t, *hidden.shape[1:])  # (b,t,nv,cv)

        dec_in_list = []
        for i in range(t):
            dec_in = self.rule_network(hidden[:, i, ...])
            dec_in_list.append(dec_in)

        dec_ins = pt.stack(dec_in_list, dim=1)
        dec_outs = self.decoder(dec_ins.flatten(0, 1))
        dec_outs = dec_outs.view(b, t, *dec_outs.shape[1:])

        return pt.sigmoid(dec_outs)

    @staticmethod
    def build_encoder(ci, hi, wi, co):
        assert (hi == 64) and (wi == 64)
        return nn.Sequential(
            nn.Conv2d(ci, 16, 5, stride=2, padding=2), nn.ELU(), LayerNorm(),  # 64->32
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ELU(), LayerNorm(),  # 32->16
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ELU(), LayerNorm(),  # 16->8
            nn.Flatten(1),
            nn.Linear(4096, co), nn.ELU(), LayerNorm()
        )

    @staticmethod
    def build_decoder(ci, hi, wi, co):
        assert (hi == 8) and (wi == 8)
        return nn.Sequential(
            LayerNorm(),
            nn.Linear(ci, 4096), nn.ELU(), LayerNorm(),
            Reshape([64, 8, 8]),
            nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.PixelShuffle(2), nn.ELU(), LayerNorm(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.PixelShuffle(2), nn.ELU(), LayerNorm(),
            nn.Conv2d(16, co * 4, 5, stride=1, padding=2), nn.PixelShuffle(2),  # nn.Sigmoid()
        )
