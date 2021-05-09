import torch
import torchvision.transforms as transforms
from tqdm.auto import tqdm
import wandb
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader

from models import DeepDANN
import options

import datetime
import logging
import os
import time

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter("[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d - %(message)s\n")

stdout_logger = logging.StreamHandler()
stdout_logger.setLevel(logging.INFO)
stdout_logger.setFormatter(fmt)
logger.addHandler(stdout_logger)

human_readable_time = datetime.datetime.fromtimestamp(time.time()).isoformat()
logfile = f"logs/log_{human_readable_time}.log"
file_logger = logging.FileHandler(logfile)
file_logger.setLevel(logging.DEBUG)
file_logger.setFormatter(fmt)
logger.addHandler(file_logger)
logger.debug(f"Logging to {logfile}")


def build_model(name, num_classes, num_domains):
    model = DeepDANN(name, num_classes, num_domains)
    return model

def get_wilds_dataset(name, data_root):
    dataset = get_dataset(dataset=name, root_dir=data_root, download=False)
    logger.info(f"Loaded dataset {name} with {len(dataset)} examples")
    return dataset

def get_split(dataset, split_name, transforms=None):
    # split name is train, val, or test
    if split_name == 'test':
        assert False, "Hi! You just tried to load a test split. This line of code is here to prevent you from shooting yourself in the foot. Comment this out to run on test."
    return dataset.get_subset(split_name, transform=transforms)
def train_step(iteration, model, train_loader, loss_class, loss_domain, optimizer, limit_batches=-1):
    model.train()
    all_class_true, all_class_pred, all_metadata, all_domain_pred = [], [], [], []
    optimizer.zero_grad()
    for i, (x, y_true, metadata) in tqdm(enumerate(train_loader)):
        if i == limit_batches:
            logger.warn(f"limit_batches set to {limit_batches}; early exit")
            break
        output = model(x) #TODO: apply mixup, but only to the domain reps?
        #mixup_criterion(loss_domain, all_domain_pred, all_metadata, output.permutation, output.lam)
        err_class = loss_class(output.logits, y_true)
        err_domain = loss_domain(output.domain_logits, metadata)
        err = err_class + err_domain
        err.backward()
        optimizer.step()
        all_class_true += y_true
        all_class_pred += output.logits
        all_metadata += metadata
        all_domain_pred += output.domain_logits
    return all_class_true, all_class_pred, all_metadata, all_domain_pred

def train(train_loader, val_loader, model, n_epochs, get_train_metrics=True, save_every=5, max_val_batches=100):
    # define loss function and optimizer
    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #
    for p in model.parameters():
        p.requires_grad = True
    # train the network
    metrics = []
    for i in range(n_epochs):
        import pdb; pdb.set_trace()
        all_class_true, all_class_pred, all_metadata, all_domain_pred = train_step(i, model, train_loader, loss_class, loss_domain, optimizer)
        if get_train_metrics:
            train_metrics = dataset.eval(all_class_pred, all_class_true, all_metadata, limit_batches=max_val_batches)
            train_metrics = metric_eval(all_class_true, all_class_pred, all_metadata, all_domain_pred)
        val_metrics = evaluate(i, model, val_loader, limit_batches=max_val_batches)
        if i % save_every == 0:
            torch.save(model.state_dict(), '~/Projects/spicy-dann') # TODO
        metrics.append(val_metrics if not get_train_metrics else (train_metrics, val_metrics))
        val_metrics = evaluate(i, model, val_loader, limit_batches=max_val_batches)
        if i % save_every == 0:
            torch.save(model.state_dict(), "./models/{run_name}_ep{i}_{human_readable_time}.ckpt")
    return metrics

def metric_eval(all_class_true, all_class_pred, all_metadata, all_domain_pred):
    class_acc = 0.0
    domain_acc = 0.0
    batch_size = all_class_true.shape[0]
    total = 0
    with torch.no_grad():
        for index in range(batch_size):
            total += all_class_true[index].shape[0]
            class_acc += [all_class_true[index]==all_class_pred[index]].sum()
            domain_acc += [all_metadata[index]==all_domain_pred[index]].sum()
        class_acc /= total
        domain_acc /= total
    metrics = (class_acc, domain_acc)
    return metrics

def evaluate(iteration, model, val_loader, limit_batches=-1):
    all_class_true, all_class_pred, all_metadata, all_domain_pred = [], [], [], []
    model.eval()
    with torch.no_grad():
        for i, (x, y_true, metadata) in tqdm(enumerate(val_loader)):
            if i == limit_batches:
                logger.warn(f"limit_batches set to {limit_batches}; early exit")
                break
        output = model(x)
        all_class_true += y_true
        all_class_pred += output.logits
        all_metadata += metadata
        all_domain_pred += output.domain_logits
    metrics = dataset.eval(all_class_pred, all_class_true, all_metadata)
    metrics = metric_eval(all_class_true, all_class_pred, all_metadata, all_domain_pred)
    return metrics

NUM_CLASSES = {
    "camelyon17": 2,
    "iwildcam": 182
}

NUM_DOMAINS = {
    "camelyon17": 3,
    "iwildcam": 243
}

DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]
)

if __name__ == '__main__':
    opts = options.get_opts()
    logger.info(f"Options:\n{options.prettyprint(opts)}")
    print(f"Loading dataset {opts.dataset} from {opts.data_root}")
    dataset = get_wilds_dataset(opts.dataset, opts.data_root)
    print('Setting up dataloader')
    train_data, val_data = get_split(dataset, 'train', transforms=DEFAULT_TRANSFORM), get_split(dataset, 'val', transforms=DEFAULT_TRANSFORM)
    train_loader, val_loader = get_train_loader('standard', train_data, batch_size=opts.batch_size), get_eval_loader('standard', val_data, batch_size=opts.batch_size)
    print(f'Build model of type {opts.model_name}')
    model = build_model(opts.model_name, NUM_CLASSES[opts.dataset], NUM_DOMAINS[opts.dataset])

    print("Configuring training")
    wandb.init(project='deep-domain-mixup', entity='tchainzzz', name=opts.run_name)
    wandb.config.update(vars(opts))
    metrics = train(train_loader, val_loader, model, opts.n_epochs, get_train_metrics=opts.get_train_metrics, save_every=opts.save_every, max_val_batches=opts.max_val_batches)
    torch.save(model.state_dict(), "./models/{opts.run_name}_final_{human_readable_time}.pth")
