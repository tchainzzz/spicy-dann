import torch
from tqdm.auto import tqdm
from wilds import get_dataset

from models import DeepDANN

import options

import datetime
import logging
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

DATA_DIR = "./WILDS"

def build_model(name, num_classes, num_domains):
    model = DeepDANN(name, num_classes, num_domains)
    return model

def get_wilds_dataset(name):
    dataset = get_dataset(dataset=name, root_dir=DATA_DIR, download=False)
    logger.info(f"Loaded dataset {name} with {len(dataset)} examples")
    return dataset

def get_split(dataset, split_name, transforms=None):
    # split name is train, val, or test
    if split_name == 'test':
        assert False, "Hi! You just tried to load a test split. This line of code is here to prevent you from shooting yourself in the foot. Comment this out to run on test."
    return dataset.get_subset(split_name, transform=transforms)

def train_step(iteration, model, train_loader, limit_batches=-1):
    model.train()
    all_y_true, all_y_pred, all_metadata = [], [], []
    for i, (x, y_true, metadata) in tqdm(enumerate(train_loader)):
        if i == limit_batches:
            logger.warn(f"limit_batches set to {limit_batches}; early exit")
            break
        y_pred = model(x) #TODO: apply mixup, but only to the domain reps?
        all_y_true += y_true
        all_y_pred += y_pred
        all_metadata += metadata
    return all_y_true, all_y_pred, all_metadata

def train(train_loader, val_loader, model, n_epochs, get_train_metrics=True, save_every=5, max_val_batches=100):
    metrics = []
    for i in range(n_epochs):
        y_true, y_pred, metadata = train_step(i, model, train_loader)
        if get_train_metrics:
            train_metrics = dataset.eval(y_pred, y_true, metadata, limit_batches=max_val_batches)
        val_metrics = evaluate(i, model, val_loader, limit_batches=max_val_batches)
        if i % save_every == 0:
            torch.save() # TODO
        metrics.append(val_metrics if not get_train_metrics else (train_metrics, val_metrics))
    return metrics

def evaluate(iteration, model, val_loader, limit_batches=-1):
    all_y_true, all_y_pred, all_metadata = [], [], []
    model.eval()
    with torch.no_grad():
        for i, (x, y_true, metadata) in tqdm(enumerate(val_loader)):
            if i == limit_batches:
                logger.warn(f"limit_batches set to {limit_batches}; early exit")
                break
        y_pred = model(x)
        all_y_true += y_true
        all_y_pred += y_pred
        all_metadata += metadata
    metrics = dataset.eval(all_y_pred, all_y_true, all_metadata)
    return metrics

NUM_CLASSES = {
    "camelyon17": 2,
    "iwildcam": 10
}
NUM_DOMAINS = {
    "camelyon17": 3,
    "iwildcam": 10
}

if __name__ == '__main__':
    opts = options.get_opts()
    logger.info(f"Options:\n{options.prettyprint(opts)}")
    dataset = get_wilds_dataset(opts.dataset)
    train_loader, val_loader = get_split(dataset, 'train'), get_split(dataset, 'val')
    model = build_model(opts.model_name, NUM_CLASSES[opts.dataset], NUM_DOMAINS[opts.dataset])
    metrics = train(train_loader, val_loader, model, opts.n_epochs, get_train_metrics=opts.get_train_metrics, save_every=opts.save_every, max_val_batches=opts.max_val_batches)

###
