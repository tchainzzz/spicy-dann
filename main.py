import torch
import torchvision.transforms as transforms
from tqdm.auto import tqdm
import wandb
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from models import DeepDANN
import options
from utils import dict_formatter

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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def build_model(name, num_classes, num_domains, domain_task_weight, mixup_param):
    model = DeepDANN(name, num_classes, num_domains, alpha=domain_task_weight, beta=mixup_param)
    return model.to(device)

def get_wilds_dataset(name, data_root):
    dataset = get_dataset(dataset=name, root_dir=data_root, download=False)
    logger.info(f"Loaded dataset {name} with {len(dataset)} examples")
    return dataset

def get_split(dataset, split_name, transforms=None):
    # split name is train, val, or test
    #if split_name == 'test':
    #    assert False, "Hi! You just tried to load a test split. This line of code is here to prevent you from shooting yourself in the foot. Comment this out to run on test."
    return dataset.get_subset(split_name, transform=transforms)
def train_step(iteration, model, train_loader, grouper, loss_class, loss_domain, optimizer, limit_batches=-1, binary=False):
    model.train()
    all_class_true, all_class_logits, all_domain_true, all_domain_logits = [], [], [], []
    optimizer.zero_grad()
    pbar = tqdm(train_loader)
    pbar.set_description(f"Epoch {iteration+1}")
    for i, (x, y_true, metadata) in enumerate(pbar):
        if i == limit_batches:
            logger.warning(f"limit_batches set to {limit_batches}; early exit")
            break
        x = x.to(device)
        y_true = y_true.to(device)
        raw_metadata = grouper.metadata_to_group(metadata).to(device)
        raw_domain = torch.unique(raw_metadata)
        domain_values = raw_domain.topk(raw_domain.numel()).indices # all domain labels are unique and can be deterministically ordered, so use topk
        domain_true = torch.zeros_like(y_true).to(device)
        for old, new in zip(raw_domain, domain_values):
            domain_true[raw_metadata == old] = new
        
        output = model(x) #TODO: apply mixup, but only to the domain reps?
        #mixup_criterion(loss_domain, all_domain_pred, all_metadata, output.permutation, output.lam)
        err_class = loss_class(output.logits, y_true)
        err_domain = loss_domain(output.domain_logits, domain_true)
        err = err_class + err_domain
        losses = {"cls/loss": err_class.item(), "dom/loss": err_domain.item(), "loss": err.item()}
        log('train', losses)

        err.backward()
        optimizer.step()
        all_class_true += y_true
        all_class_logits += output.logits
        all_domain_true += domain_true
        all_domain_logits += output.domain_logits

        class_preds = torch.max(output.logits, dim=-1).indices
        domain_preds = torch.max(output.domain_logits, dim=-1).indices
        train_metrics = compute_metrics(y_true, class_preds, domain_true, domain_preds, binary=binary)
        log('train', train_metrics)
        logger.debug(f"Logging validation metrics for epoch {iteration+1}, batch {i+1} of {len(train_loader)}: {dict_formatter(train_metrics)}")
        pbar.set_postfix(train_metrics)
    return all_class_true, all_class_logits, all_domain_true, all_domain_logits


def train(train_loader, val_loader, model, grouper, n_epochs, get_train_metrics=True, save_every=5, max_val_batches=100, binary=False):
    # define loss function and optimizer # TODO: command-line-ify this
    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #
    for p in model.parameters():
        p.requires_grad = True
    # train the network
    metrics = []
    for i in range(n_epochs):
        all_class_true, all_class_pred, all_domain_true, all_domain_pred = train_step(i,
            model, train_loader, grouper,
            loss_class, loss_domain, optimizer, limit_batches=max_val_batches, binary=binary)
        val_metrics = evaluate(i, val_loader, model, grouper, limit_batches=max_val_batches, binary=binary)
        log('val', val_metrics)
        print(f"Validation metrics: {dict_formatter(val_metrics)}")
        logger.debug(f"Logging validation metrics for epoch {i+1}: {dict_formatter(val_metrics)}")
        metrics.append(val_metrics)
        if i % save_every == 0:
            torch.save(model.state_dict(), "./models/{run_name}_ep{i}_{human_readable_time}.ckpt")
    return metrics


def compute_metrics(all_class_true, all_class_pred, all_domain_true=None, all_domain_pred=None, binary=False):
    all_class_true = all_class_true.cpu().numpy()
    all_class_pred = all_class_pred.cpu().numpy()
    class_acc = accuracy_score(all_class_true, all_class_pred)
    class_prec, class_rec, class_f1, *_ = precision_recall_fscore_support(all_class_true, all_class_pred, average='binary' if binary else 'macro')
    metrics = {
        "cls/acc": class_acc,
        "cls/prec": class_prec,
        "cls/rec":class_rec,
        "cls/f1": class_f1,

    }

    if all_domain_true is not None and all_domain_pred is not None:
        all_domain_true = all_domain_true.cpu().numpy()
        all_domain_pred = all_domain_pred.cpu().numpy()
        domain_acc = accuracy_score(all_domain_true, all_domain_pred)
        domain_prec, domain_rec, domain_f1, *_ = precision_recall_fscore_support(all_class_true, all_class_pred, average='macro')
        domain_metrics = {
            "dom/acc": domain_acc,
            "dom/prec": domain_prec,
            "dom/rec": domain_rec,
            "dom/f1": domain_f1
        }
        metrics.update(domain_metrics)
    return metrics

def evaluate(iteration, val_loader, model, grouper, limit_batches=-1, binary=False):
    all_class_true, all_class_pred = [], []

    loss_class = torch.nn.NLLLoss()
    model.eval()
    err_class = 0
    with torch.no_grad():
        pbar = tqdm(val_loader)
        pbar.set_description(f"Validating epoch {iteration+1}")
        for i, (x, y_true, metadata) in enumerate(pbar):
            if i == limit_batches:
                logger.warning(f"limit_batches set to {limit_batches}; early exit")
                break
            x = x.to(device)
            y_true = y_true.to(device)
            output = model(x)
            err_class += loss_class(output.logits, y_true)
            all_class_true += y_true
            class_preds = torch.max(output.logits, dim=-1).indices
            all_class_pred += class_preds
    log('val', {"cls/loss": err_class.item() / len(val_loader)})
    metrics = compute_metrics(
        torch.stack(all_class_true, dim=0),
        torch.stack(all_class_pred, dim=0),
        binary=binary
    )
    return metrics

def log(split, metric_dict):
    for name, value in metric_dict.items():
        wandb.log({f"{split}/{name}": value})


NUM_CLASSES = {
    "camelyon17": 2,
    "iwildcam": 182
}

NUM_DOMAINS = {
    "camelyon17": 3,
    "iwildcam": 243
}

NUM_VAL_DOMAINS = {
    "camelyon17": 1,
    "iwildcam": 32
}

NUM_TEST_DOMAINS = {
    "camelyon17": 1,
    "iwildcam": 48
}

METADATA_KEYS = { # what domain SHIFT are we trying to model?
    "camelyon17": "hospital",
    "iwildcam": "location"
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
    train_data = get_split(dataset, 'train', transforms=DEFAULT_TRANSFORM)
    val_data = get_split(dataset, 'test', transforms=DEFAULT_TRANSFORM)

    grouper = CombinatorialGrouper(dataset, [METADATA_KEYS[opts.dataset]])
    train_loader = get_train_loader('group', train_data, batch_size=opts.batch_size, grouper=grouper, n_groups_per_batch=min(opts.batch_size, NUM_DOMAINS[opts.dataset]))
    val_loader = get_eval_loader('standard', val_data, batch_size=opts.batch_size) # we don't care about test-time domain class.
    
    assert train_loader is not None
    assert val_loader is not None
    print(f'Build model of type {opts.model_name}')
    model = build_model(opts.model_name, NUM_CLASSES[opts.dataset], NUM_DOMAINS[opts.dataset], opts.domain_task_weight, opts.mixup_param)

    print("Configuring training")
    wandb.init(project='deep-domain-mixup', entity='tchainzzz', name=opts.run_name)
    wandb.config.update(vars(opts))

    metrics = train(train_loader, val_loader, model, grouper, opts.n_epochs, 
        get_train_metrics=opts.get_train_metrics, save_every=opts.save_every, 
        max_val_batches=opts.max_val_batches, binary=(opts.dataset == 'camelyon17'))
    torch.save(model.state_dict(), "./models/{opts.run_name}_final_{human_readable_time}.pth")
