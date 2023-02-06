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
            a = 1. / np.sqrt(dout)
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


class SequentialArithmeticNps(nn.Module):

    def __init__(self,
            nr, cr, nv, cv
    ):
        super().__init__()
        self.encoder_operand = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, cv))
        self.encoder_operator = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, cv))
        # self.decoder_operand = MlpL2(cv, 1, cm=64)  # TODO

        self.rules_body = nn.Parameter(pt.randn(nr, cr))
        self.rules_head = nn.Sequential(GLinear(2 * cv, 128, nr), nn.ReLU(), GLinear(128, 1, nr))

        self.selector1 = SelectAttention(cv, cr, 32, nq=3, nk=nr, share_q=True, share_k=True)
        self.selector2 = SelectAttention(cv, cv, 16, nq=1, nk=2, share_q=True, share_k=False)  # TODO XXX cr+cv

        print(self)

        # self.vars = None  # TODO XXX

        self.rule_selection = []
        self.primary_selection = []
        self.context_selection = []

    def forward(self, operand1, operand2, operator):  # (20,1) (20,1) (20,3)
        x1c = self.wrap_operand(operand1, 0)
        x2c = self.wrap_operand(operand2, 1)
        opc = self.wrap_operator(operator)
        x1e = self.encoder_operand(x1c)  # (20,64)
        x2e = self.encoder_operand(x2c)  # (20,64)
        ope = self.encoder_operator(opc)  # (20,64)

        hidden = pt.stack([x1e, x2e, ope], dim=1)

        b = hidden.size(0)

        q1 = hidden
        k1 = self.rules_body[None, :, :].repeat(b, 1, 1)
        attent1 = self.selector1(q1, k1)  # (1,nv,nr)
        shape_a1 = attent1.shape
        attent1_ = attent1.flatten(1)
        if self.training:
            mask1_ = ptnf.gumbel_softmax(attent1_, tau=1, hard=True, dim=1)
        else:
            mask1_ = argmax_onehot(attent1_, dim=1)
        mask1 = mask1_.view(*shape_a1)

        mask_r = mask1.sum(dim=1)  # (b,nr)
        rule_body = (self.rules_body[None, :, :] * mask_r[:, :, None]).sum(dim=1)  # (1,nr,cr),(b,nr,1)->(b,cr)
        mask_p = mask1.sum(dim=2)  # (b,nv)
        var_p = (hidden * mask_p[:, :, None]).sum(dim=1)  # (b,nv,cv),(b,nv,1)->(b,cv)

        self.rule_selection.append(pt.argmax(mask_r.detach(), dim=1).cpu().numpy())
        self.primary_selection.append(pt.argmax(mask_p.detach(), dim=1).cpu().numpy())

        hidden = hidden[:, :2, :]  # (b,2,cv) - (b,nv,cv)  # TODO remove this

        q2 = var_p[:, None, :]  # TODO XXX pt.cat([rule_body, var_p], dim=1)[:, None, :]  # (b,1,cr+cv)
        k2 = hidden
        attent2 = self.selector2(q2, k2)[:, 0, :]  # (b,1,cr+cv),(b,2,cv)->(b,2) - (b,nv)
        if self.training:
            mask2 = ptnf.gumbel_softmax(attent2, tau=1.0, hard=True, dim=1)  # (b,2) - (b,nv)
        else:
            mask2 = argmax_onehot(attent2, dim=1)

        var_c = (hidden * mask2[:, :, None]).sum(dim=1)  # (b,cv)

        self.context_selection.append(pt.argmax(mask2.detach(), dim=1).cpu().numpy())

        var_pc = pt.cat([var_p, var_c], dim=1)  # (b,cv*2)
        output = self.rules_head(var_pc[:, None, :].repeat(1, 3, 1))  # (b,nr,co)  # TODO 需要repeat吗？或者修改GLinear？
        output = (output * mask_r[:, :, None]).sum(dim=1)  # (b,co)

        return output[:, 0]

    def reset(self):
        self.rule_selection.clear()
        self.primary_selection.clear()
        self.context_selection.clear()

    @staticmethod
    def wrap_operand(xi, flag):
        """
        :param xi: in shape (b,)
        :param flag: can be set to 0,1, repesenting operand 1, 2 respectively
        :return: in shape (b,2)
        """
        assert len(xi.shape) == 1
        assert flag in (0, 1)
        extra = pt.ones_like(xi) * flag
        xo = pt.stack([xi, extra], dim=1)
        return xo

    @staticmethod
    def peel_operand(xi):
        """
        :param xi: in shape (b,4)
        :return: in shape (b,)  # and int
        """
        assert len(xi.shape) == 2 and xi.size(1) == 4
        xo = xi[:, 0]
        return xo

    @staticmethod
    def wrap_operator(xi):
        """
        :param xi: in shape (b,)
        :return: in shape (b,3)
        """
        assert len(xi.shape) == 1
        assert pt.all(pt.max(xi) < 3)
        xo = ptnf.one_hot(xi.long(), 3).to(xi.dtype)
        return xo

    @staticmethod
    def peel_operator():
        raise NotImplemented
