import torch
import torch.nn as nn

import ast

class DenseCrossEntropyLoss(nn.Module):
    def __init__(self, dim=-1, weight=None, reduction='mean'):
        super(DenseCrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.weight = weight
        self.dim = dim

    def forward(self, pred, target):
        assert pred.size() == target.size(), f"Preds and targets have different sizes: {pred.size()} {target.size()}"
        pred = pred.log_softmax(dim=self.dim)
        if self.weight is not None:
            pred = pred * self.weight.unsqueeze(0)   
        loss = torch.sum(-target * pred, dim=self.dim)
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return torch.sum(loss)

def dict_formatter(d, delimiter=":", joiner="-"):
    return f" {joiner} ".join([f"{k}{delimiter} {v:.4f}" for k, v in d.items()])


def parse_argdict(argdict):
    result = {}
    for kv in argdict:
        key, val = kv.split("=")
        try:
            result[key] = ast.literal_eval(val)
        except ValueError:
            result[key] = str(val)
    return result


