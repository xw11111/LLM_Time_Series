import torch
import torch.nn as nn
from typing import Optional, Sequence, Union

Floats = Union[float, Sequence[float]]

def listify(p):
    return list(p) if isinstance(p, (list, tuple)) else [p]

def bn_drop_lin(ni: int, no: int, bn: bool=True, p: float=0., act=None):
    layers = []
    if bn: layers.append(nn.BatchNorm1d(ni))
    if p and p > 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(ni, no))
    if act is not None: layers.append(act)
    return nn.Sequential(*layers)

class Flatten(nn.Module):
    def forward(self, x): return x.contiguous().view(x.size(0), -1)

class AdaptiveConcatPool1d(nn.Module):
    def __init__(self, sz: Optional[int]=None):
        super().__init__()
        sz = sz or 1
        self.ap, self.mp = nn.AdaptiveAvgPool1d(sz), nn.AdaptiveMaxPool1d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], dim=1)

def create_head1d(nf: int, nc: int,
                  lin_ftrs: Optional[Sequence[int]] = None,
                  ps: Floats = 0.5, bn_final: bool = False, bn: bool = True,
                  act: str = "relu", concat_pooling: bool = True):

    first = 2*nf if concat_pooling else nf
    lin_ftrs = [first, nc] if lin_ftrs is None else [first] + list(lin_ftrs) + [nc]

    ps = listify(ps)
    if len(ps) == 1:

        ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps

    act_mod = nn.ReLU(inplace=True) if act == "relu" else nn.ELU(inplace=True)
    actns = [act_mod] * (len(lin_ftrs)-2) + [None]

    layers = [AdaptiveConcatPool1d() if concat_pooling else nn.MaxPool1d(2), Flatten()]
    for ni, no, p, a in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers.append(bn_drop_lin(ni, no, bn=bn, p=p, act=a))
    if bn_final:
        layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)