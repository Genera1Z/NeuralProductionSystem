from functools import partial
from typing import Iterable, Optional, Union

import torch
from torch import nn, Tensor

InitFn = partial


def uniform(mean: float = 0., var: float = 1.) -> InitFn:
    return partial(nn.init.uniform_, a=mean, b=var)


def orthogonal(gain: float = 1.) -> InitFn:
    return partial(nn.init.orthogonal_, gain=gain)


def kaiming_normal(nonlinearity: str = 'relu') -> InitFn:
    return partial(nn.init.kaiming_normal_, nonlinearity=nonlinearity)


def kaiming_uniform(nonlinearity: str = 'relu') -> InitFn:
    return partial(nn.init.kaiming_uniform_, nonlinearity=nonlinearity)


def constant(val: float) -> InitFn:
    return partial(nn.init.constant_, val=val)


def zero() -> InitFn:
    return partial(nn.init.constant_, val=0)


def lstm_bias(forget: float = 1.0, other: float = 0.0) -> InitFn:
    """Set forget bias and others separately.
    """

    def __set_bias(t: Tensor) -> None:
        with torch.no_grad():
            i = len(t) // 4
            t.fill_(other)
            t[i:2 * i].fill_(forget)

    return partial(__set_bias)


class Initializer:
    """Utility Class to initialize weight parameters of NN
    """

    def __init__(
            self,
            nonlinearity: Optional[str] = None,
            weight_init: InitFn = orthogonal(),
            bias_init: InitFn = zero(),
            scale: float = 1.,
    ) -> None:
        """If nonlinearity is specified, use orthogonal
           with calucurated gain by torch.init.calculate_gain.
        """
        self.weight_init = weight_init
        if nonlinearity is not None:
            if 'gain' in self.weight_init.keywords:
                self.weight_init.keywords['gain'] = nn.init.calculate_gain(nonlinearity)
            elif 'nonlinearity' in self.weight_init.keywords:
                self.weight_init.keywords['nonlinearity'] = nonlinearity
            else:
                raise ValueError('{} doesn\'t have gain', self.weight_init)
        self.bias_init = bias_init
        self.scale = scale

    def __call__(self, mod: Union[nn.Module, nn.Sequential, Iterable[nn.Module]]) -> nn.Module:
        return self.__init_dispatch(mod)

    def make_list(self, *args) -> nn.ModuleList:
        return nn.ModuleList([self.__init_dispatch(mod) for mod in args])

    def make_seq(self, *args) -> nn.Sequential:
        return nn.Sequential(*map(lambda mod: self.__init_dispatch(mod), args))

    def __init_dispatch(self, mod: nn.Module) -> nn.Module:
        if isinstance(mod, nn.Sequential) or isinstance(mod, nn.ModuleList):
            for child in mod.children():
                self.__init_dispatch(child)
        else:
            self.__init_mod(mod)
        return mod

    def __init_mod(self, mod: nn.Module) -> nn.Module:
        if isinstance(mod, nn.BatchNorm2d):
            return self.__init_batch_norm(mod)
        for name, param in mod.named_parameters():
            if 'weight' in name:
                self.weight_init(param)
            elif 'bias' in name:
                self.bias_init(param)
        return mod

    def __init_batch_norm(self, mod: nn.BatchNorm2d) -> nn.Module:
        mod.weight.data.fill_(1)
        mod.bias.data.zero_()
        return mod
