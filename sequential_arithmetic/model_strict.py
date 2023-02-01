import numpy as np
import torch as pt
import torch.autograd as pta
import torch.nn as nn
import torch.nn.functional as ptnf


class BatchArgMax(pta.Function):

    @staticmethod
    def jvp(ctx, *grad_inputs):
        pass

    @staticmethod
    def forward(ctx, *args, **kwargs):
        x, = args
        idx0 = pt.argmax(x.flatten(1), dim=1)
        idx_list = []
        for i, d0 in enumerate(x.shape[1:]):
            base = np.prod(x.shape[i + 2:])
            di = idx0 // base
            idx_list.append(di)
            idx0 = idx0 % base
        idx = pt.stack(idx_list, 1).to(x.device)  # XXX must not use ``.long()``  # (b,len(x.shape)-1)
        ctx.save_for_backward(x, idx.long())
        return idx

    @staticmethod
    def backward(ctx, *args, **kwargs):
        grad_idx, = args
        x, idx = ctx.saved_tensors
        grad_x = pt.zeros_like(x)
        b = pt.arange(x.shape[0])
        assert len(grad_idx.shape) == 2
        grad_m = pt.mean(grad_idx, dim=1)  # ~``sum``  # (b,)
        if idx.shape[1] == 1:
            grad_x[b, idx[:, 0]] = grad_m * x[b, idx[:, 0]]  # without ``* x[]`` is a bit worse
        elif idx.shape[1] == 2:
            grad_x[b, idx[:, 0], idx[:, 1]] = grad_m * x[b, idx[:, 0], idx[:, 1]]
        else:  # elif idx.shape[1] == 3|4|5...
            raise NotImplemented
        return grad_x


class BatchSelect(pta.Function):

    @staticmethod
    def jvp(ctx, *grad_inputs):
        pass

    @staticmethod
    def forward(ctx, *args, dim=1, **kwargs):
        x, idx = args
        assert dim == 1
        assert len(idx.shape) == 1  # idx: (b,)
        assert idx.shape[0] == x.shape[0]
        idx = idx.long()
        ctx.save_for_backward(x, idx)
        b = pt.arange(x.shape[0])
        return x[b, idx, ...]  # (b,n,...) -> (b,...)

    @staticmethod
    def backward(ctx, *args, **kwargs):
        grad_select, = args
        x, idx = ctx.saved_tensors
        assert len(grad_select.shape) == 2
        grad_idx = pt.mean(grad_select, dim=1)  # ~``sum``; ``ones_like(idx)`` is much worse  # (b,)
        grad_x = pt.zeros_like(x)
        b = pt.arange(x.shape[0])
        grad_x[b, idx, ...] = grad_select
        return grad_x, grad_idx


bargmax = BatchArgMax.apply
bselect = BatchSelect.apply


# class GLinear(nn.Module):
#
#     def __init__(self,
#             din, dout, num_blocks, bias=True, a=None
#     ):
#         super(GLinear, self).__init__()
#         if a is None:
#             a = 1. / np.sqrt(dout)
#         self.weight = nn.Parameter(pt.FloatTensor(num_blocks, din, dout).uniform_(-a, a))
#         self.bias = bias
#         if bias is True:
#             self.bias = nn.Parameter(pt.FloatTensor(num_blocks, dout).uniform_(-a, a))
#         else:
#             self.bias = None
#
#     def forward(self, x):
#         x = x.permute(1, 0, 2)
#         x = pt.bmm(x, self.weight)
#         x = x.permute(1, 0, 2)
#         if not self.bias is None:
#             x = x + self.bias
#         return x


class MlpL2(nn.Sequential):

    def __init__(self, ci, co, cm=32):
        super(MlpL2, self).__init__(
            nn.Linear(ci, cm), nn.ReLU(), nn.Linear(cm, co)
        )


class RuleSet(nn.Module):

    def __init__(self,
            n, cr, cv, cm=128
    ):
        super(RuleSet, self).__init__()
        self.n = n
        self.head = nn.ModuleList([
            nn.Sequential(nn.Linear(cv * 2, cm), nn.ReLU(), nn.Linear(cm, cv)) for _ in range(n)
        ])
        self.body = nn.Parameter(pt.randn(n, cr), requires_grad=True)  # XXX ``self.body``
        # self._body_proj = nn.ModuleList([  # TODO XXX a bit better
        #     nn.Linear(cr, cr) for _ in range(n)
        # ])
        # self._body_proj = nn.Sequential(
        #     nn.Linear(cr, cr // 2), nn.ReLU(), nn.Linear(cr // 2, cr)
        # )

    # @property
    # def body(self):
    #     return pt.concat([self._body_proj[_](self._body[None, _]) for _ in range(self._body.size(0))])
    #     # return self._body_proj(self._body)

    def forward(self, xis, idxs_b):
        """
        :param xis: in shape (b,ci)
        :param idxs_b: in shape (b,)
        :return: in shape (b,co)
        """
        xo_list = [0] * xis.size(0)
        for idx_h in range(self.n):
            with pt.no_grad():
                idxs_x = pt.where(idxs_b == idx_h)[0].cpu().numpy()
                if len(idxs_x) == 0:
                    continue
            xos_ = self.head[idx_h](xis[idxs_x])
            for idx, xo in zip(idxs_x, xos_):
                # with pt.no_grad():
                xo_list[idx] = xo
        xos = pt.stack(xo_list, 0)
        return xos

    def body_repeat(self, b):
        return self.body[None, :, :].repeat(b, 1, 1)


class QkAttention(nn.Module):

    def __init__(self, cq, ck, cm=16):
        super(QkAttention, self).__init__()
        self.proj_q = nn.Linear(cq, cm)
        self.proj_k = nn.Linear(ck, cm)
        self.temperature = np.sqrt(cm)

    def forward(self, q, k):
        """
        :param q: in shape (b,nq,cq)
        :param k: in shape (b,nk,ck)
        :return: in shape (b,nq,nk)
        """
        read = self.proj_q(q)
        write = self.proj_k(k)
        attent = pt.bmm(read, write.permute(0, 2, 1)) / self.temperature
        return attent


# class SelectAttention(nn.Module):
#
#     def __init__(self,
#             cq, ck, cm=16, nq=5, nk=5, share_q=False, share_k=False
#     ):
#         super(SelectAttention, self).__init__()
#
#         if not share_q:
#             self.proj_q = GLinear(cq, cm, nq)
#         else:  # V
#             self.proj_q = nn.Linear(cq, cm)
#
#         if not share_k:  # V
#             self.proj_k = GLinear(ck, cm, nk)
#         else:
#             self.proj_k = nn.Linear(ck, cm)
#
#         self.temperature = np.sqrt(cm)  # 32^0.5
#
#     def forward(self, q, k):
#         read = self.proj_q(q)
#         write = self.proj_k(k)
#         attent = pt.bmm(read, write.permute(0, 2, 1)) / self.temperature
#         return attent


class ArithmeticNps(nn.Module):

    def __init__(self,
            cv, n_rule, cr
    ):
        super(ArithmeticNps, self).__init__()
        # self.encoder_operand = MlpL2(4, hidd_dim, 64)
        # self.encoder_operator = MlpL2(3, hidd_dim, 64)
        # self.decoder_operand = MlpL2(hidd_dim, 4, 64)
        # self.decoder_operator = MlpL2(hidd_dim, 3, 64)
        self.encoder_operand = MlpL2(2, cv, cm=64)
        self.encoder_operator = MlpL2(3, cv, cm=64)
        self.decoder_operand = MlpL2(cv, 1, cm=64)

        self.state_encoder = MlpL2(cv * 2, cv, cm=64)

        self.rules = RuleSet(n_rule, cr, cv, 128)

        self.selector1 = QkAttention(cr, cv, cm=32)
        self.selector2 = QkAttention(cr, cv, cm=16)
        # self.selector1 = SelectAttention(
        #     cr, cv, cm=32, nq=n_rule, nk=3, share_q=True, share_k=False
        # )
        # self.selector2 = SelectAttention(
        #     cr, cv, cm=16, nq=1, nk=2, share_q=False, share_k=False
        # )

        self.state = None
        self.rule_selection = []

        self.tau1 = 1
        self.tau2 = 1

    def forward(self, operand1, operand2, operator):
        # def forward(self, x1c, x2c, opc):
        # def forward(self, x1e, x2e, ope):
        """
        :param operand1: in shape (b,)
        :param operand2: in shape (b,)
        :param operator: in shape (b,)
        :return: in shape (b,)
        """
        x1c = self.wrap_operand(operand1, 0)
        x2c = self.wrap_operand(operand2, 1)
        opc = self.wrap_operator(operator)
        x1e = self.encoder_operand(x1c)
        x2e = self.encoder_operand(x2c)
        ope = self.encoder_operator(opc)

        hidden = pt.stack([x1e, x2e, ope], dim=1)
        b = hidden.size(0)

        pta.set_detect_anomaly(True)
        if self.state is None:  # TODO XXX
            self.state = pt.zeros_like(hidden)
        hidden = self.state_encoder(pt.concat([self.state, hidden], dim=2))
        self.state = hidden

        attent1 = self.selector1(self.rules.body_repeat(b), hidden)  # (b,nr,nv)

        shape_a1 = attent1.shape
        attent1_ = attent1.flatten(1)
        if self.training:
            prob1_ = ptnf.gumbel_softmax(attent1_, tau=self.tau1, hard=True, dim=1)
        else:
            prob1_ = ptnf.softmax(attent1_, dim=1)
        # self.rule_probs.append(prob1_.detach().view(shape_a1))

        prob1 = prob1_.view(*shape_a1)
        assert len(prob1.shape) == 3

        idx_rp = bargmax(prob1)  # (b,nr,nv) -> (b,2): [[idx_nr,idx_nv],..]
        idx_r, idx_p = [idx_rp[:, _] for _ in range(2)]  # (b,) (b,)
        rule_body = bselect(self.rules.body_repeat(b), idx_r)  # (b,cr)
        var_p = bselect(hidden, idx_p)  # (b,cv)

        self.rule_selection.append(idx_r.detach().cpu().numpy())

        hidden = hidden[:, :2]  # TODO 去掉
        attent2 = self.selector2(rule_body[:, None, :], hidden)[:, 0, :]  # (b,1,cr)|(b,nv,cv) -> (b,1,nv) -> (b,nv)

        if self.training:
            prob2 = ptnf.gumbel_softmax(attent2, tau=self.tau2, hard=True, dim=1)
        else:
            prob2 = ptnf.softmax(attent2, dim=1)
        assert len(prob2.shape) == 2

        idx_c = bargmax(prob2)[:, 0]  # (b,nv) -> (b,1) -> (b,): [idx_nv,..]
        var_c = bselect(hidden, idx_c)  # (b,cv)

        var_pc = pt.cat([var_p, var_c], dim=1)  # (b,cv*2)
        var_p = self.rules(var_pc, idx_r)  # (b,cv)

        # self.state[pt.arange(b), idx_p.long()] += var_p  # TODO XXX
        new_state = self.state.detach().clone()  # TODO XXX
        new_state[pt.arange(b), idx_p.long()] += var_p.detach()
        self.state = new_state

        x3c = self.decoder_operand(var_p)[:, 0]

        # result = self.peel_operand(x3c)
        # if self.training:
        #     return result, x3c
        # else:
        #     return result
        return x3c
        # return var_p

    def reset(self):
        self.rule_selection.clear()
        self.state = None

    @staticmethod
    def wrap_operand(xi, flag):
        # """
        # :param xi: in shape (b,)
        # :param flag: can be set to 0,1,2, repesenting operand 1, 2 and result respectively
        # :return: in shape (b,3)
        # """
        # assert len(xi.shape) == 1
        # assert flag in (0, 1, 2)
        # extra = pt.tensor([flag] * xi.size(0)).to(xi.device)
        # extra = ptnf.one_hot(extra, 3).to(xi.dtype)
        # xo = pt.concat([xi[:, None], extra], dim=1)
        # return xo
        assert len(xi.shape) == 1
        assert flag in (0, 1)
        extra = pt.ones_like(xi) * flag
        xo = pt.stack([xi, extra], dim=1)
        return xo

    @staticmethod
    def peel_operand(xi):
        """
        :param xi: in shape (b,3)
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
