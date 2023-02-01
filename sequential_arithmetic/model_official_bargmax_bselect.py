import math

import torch as pt
import torch.autograd as pta
import torch.nn as nn
import torch.nn.functional as ptnf

from model_strict import bargmax, bselect, RuleSet


class ArgMax(pta.Function):

    @staticmethod
    def forward(ctx, input):
        idx = pt.argmax(input, 1)
        op = pt.zeros_like(input)
        op.scatter_(1, idx[:, None], 1)
        ctx.save_for_backward(op)
        return op

    @staticmethod
    def backward(ctx, grad_output):
        op, = ctx.saved_tensors
        grad_input = grad_output * op
        return grad_input


class GLinear(nn.Module):

    def __init__(self,
            din, dout, num_blocks, bias=True, a=None
    ):
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


class GMlpL2(nn.Sequential):

    def __init__(self,
            ci, co, num, cm=128
    ):
        super(GMlpL2, self).__init__(
            GLinear(ci, cm, num), nn.ReLU(), GLinear(cm, co, num)
        )


class MlpL2(nn.Sequential):

    def __init__(self, ci, co, cm=32):
        super(MlpL2, self).__init__(
            nn.Linear(ci, cm), nn.ReLU(), nn.Linear(cm, co)
        )


class SelectAttention(nn.Module):

    def __init__(self,
            cq, ck, cm=16, nq=5, nk=5, share_q=False, share_k=False
    ):
        super(SelectAttention, self).__init__()
        if not share_q:  # V
            self.proj_q = GLinear(ck, cm, nk)
        else:
            self.proj_q = nn.Linear(ck, cm)

        if not share_k:
            self.proj_k = GLinear(cq, cm, nq)
        else:  # V
            self.proj_k = nn.Linear(cq, cm)

        self.temperature = math.sqrt(cm)  # 32^0.5

    def forward(self, q, k):
        read = self.proj_k(q)
        write = self.proj_q(k)
        attent = pt.bmm(read, write.permute(0, 2, 1)) / self.temperature
        return attent


class ArithmeticNps(nn.Module):

    def __init__(self,
            cv, n_rule, cr
    ):
        super().__init__()
        self.encoder_operand = MlpL2(2, cv, cm=64)
        self.encoder_operator = MlpL2(3, cv, cm=64)
        self.decoder_operand = MlpL2(cv, 1, cm=64)  # XXX 用之效果更好，但也明显更不稳定。

        self.rules = RuleSet(n_rule, cr, cv, 128)

        self.selector1 = SelectAttention(cr, cv, 32, nq=n_rule, nk=3, share_q=True, share_k=False)
        self.selector2 = SelectAttention(cr, cv, 16, nq=2, nk=2, share_q=False, share_k=False)

        # self.state = None
        self.rule_selection = []

    def forward(self, operand1, operand2, operator):  # (20,1) (20,1) (20,3)
        x1c = self.wrap_operand(operand1, 0)
        x2c = self.wrap_operand(operand2, 1)
        opc = self.wrap_operator(operator)
        x1e = self.encoder_operand(x1c)  # (20,64)
        x2e = self.encoder_operand(x2c)  # (20,64)
        ope = self.encoder_operator(opc)  # (20,64)

        hidden = pt.stack([x1e, x2e, ope], dim=1)
        b = hidden.size(0)

        attent1 = self.selector1(self.rules.body_repeat(b), hidden)  # (b,nr,nv)

        shape_a1 = attent1.shape
        attent1_ = attent1.flatten(1)
        if self.training:
            prob1_ = ptnf.gumbel_softmax(attent1_, tau=1, hard=True, dim=1)
            mask = prob1_.view(shape_a1)
        else:
            prob1_ = ptnf.softmax(attent1_, dim=1)
            mask = ArgMax().apply(prob1_).view(shape_a1)

        rule_mask = mask.sum(dim=2)  # (b,nr)
        idx_r = bargmax(rule_mask)[:, 0]
        rule_body = bselect(self.rules.body_repeat(b), idx_r)
        assert len(rule_body.shape) == 2  # TODO comment it

        self.rule_selection.append(idx_r.detach().cpu().numpy())

        hidden = hidden[:, :2, :]  # (b,2,cv)
        attent2 = self.selector2(rule_body[:, None, :].repeat(1, 2, 1), hidden)  # (b,2,2)

        if self.training:
            prob2 = ptnf.gumbel_softmax(attent2, tau=1.0, hard=True, dim=2)  # (20,2,2)
            mask21, mask22 = [prob2[:, _, :] for _ in range(2)]
        else:
            prob2 = ptnf.softmax(attent2, dim=2)
            mask21, mask22 = [ArgMax().apply(prob2[:, _, :]) for _ in range(2)]
        """
        SEQUENCE LENGTH	|	MSE
	      10        |	0.0002
	      20        |	0.0007
	      30        |	0.0013
	      40        |	0.0018
	      50        |	0.0027
        """

        assert len(mask21.shape) == 2
        assert len(mask22.shape) == 2
        idx_p = bargmax(mask21)[:, 0]
        var_p = bselect(hidden, idx_p)
        idx_c = bargmax(mask22)[:, 0]
        var_c = bselect(hidden, idx_c)

        var_pc = pt.cat([var_p, var_c], dim=1)  # (b,nv*2)
        output = self.rules(var_pc, idx_r)

        x3c = self.decoder_operand(output)[:, 0]

        return x3c

    def reset(self):
        self.rule_selection = []
        # self.variable_activation = []
        # self.variable_activation_1 = []
        # self.rule_probabilities = []
        # self.variable_probabilities = []

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
