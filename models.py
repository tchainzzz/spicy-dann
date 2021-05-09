import numpy as np
import torch
import torch.nn as nn
import torchvision

from dann.functions import ReverseLayerF
from mixup import mixup_data, mixup_criterion

from dataclasses import dataclass

@dataclass
class MixupModelOutput:
    logits: torch.Tensor
    domain_logits: torch.Tensor = None
    permutation: torch.Tensor = None
    lam: float = None


class MixUpDomainClassifier(nn.Module):
    def __init__(self, in_size, n_classes, fc_size=(1024, 1024), alpha=0.1, beta=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha # domain class. weight
        self.beta = beta # mixup distribution
        self.layers = nn.ModuleList()
        for i, size in enumerate(fc_size):
            in_features = fc_size[i - 1] if i > 0 else in_size
            out_features = fc_size[i]
            self.layers.append(nn.Linear(in_features, out_features))
        self.layers.append(nn.Linear(fc_size[-1], n_classes))

    def forward(self, X):
        if self.beta > 0:
            lam = np.random.beta(self.beta, self.beta)
        else:
            lam = 1
        X, permutation, lam = mixup_data(X, lam)
        X = ReverseLayerF.apply(X, self.alpha)
        for layer in self.layers:
            X = layer(X)
        return X, permutation, lam

class DeepDANN(nn.Module):
    def __init__(self, model_name, num_classes, num_domains, alpha=0.1, beta=-1, pretrained=True, device='cuda', *args, **kwargs):
        super().__init__(*args, **kwargs)
        model = getattr(torchvision.models, model_name)(pretrained=pretrained).to(device)
        self.alpha = alpha
        self.beta = beta
        if model_name.startswith('resnet'):
            num_ftrs = model.fc.in_features
            self.classifier = nn.Linear(num_ftrs, num_classes).to(device)
            model.fc = Identity().to(device)
            self.feature_extractor = model
            self.domain_discriminator = MixUpDomainClassifier(num_ftrs, num_domains, alpha=alpha, beta=beta).to(device)
        elif model_name.startswith('densenet'):
            num_ftrs = model.classifier.in_features
            self.classifier = nn.Linear(num_ftrs, num_classes).to(device)
            model.classifier = Identity()
            self.feature_extractor = model.to(device)
            self.domain_discriminator = MixUpDomainClassifier(num_ftrs, num_domains, alpha=alpha, beta=beta).to(device)
        else:
            raise NotImplementedError()

    def forward(self, X):
        feats = self.feature_extractor(X)
        logits = self.classifier(feats)
        if self.training:
            domain_logits, permutation, lam = self.domain_discriminator(feats)
            return MixupModelOutput(logits, domain_logits, permutation, lam)
        else:
            return MixupModelOutput(logits)
class Identity(nn.Module): # utility for deleting layers
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x