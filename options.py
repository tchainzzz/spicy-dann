from argparse import ArgumentParser
import os

def get_opts():
    psr = ArgumentParser()

    # data
    psr.add_argument("--dataset", required=True, type=str, choices=['camelyon17', 'iwildcam'])
    psr.add_argument("--data-root", required=True, type=str)
    psr.add_argument("--num-workers", default=os.cpu_count(), type=int)

    # training
    psr.add_argument("--model-name", type=str, required=True)
    psr.add_argument("--batch-size", default=16, type=int)
    psr.add_argument("--n-epochs", default=50, type=int)
    psr.add_argument("--opt-name", default="SGD", type=str)
    psr.add_argument("--lr", type=float, required=True)
    psr.add_argument("--opt-kwargs", type=str, nargs='+', default={})

    psr.add_argument("--domain-task-weight", default=1., type=float)
    psr.add_argument("--mixup-param", default=-1, type=float)

    # checkpointing
    psr.add_argument("--save-every", default=1, type=int)
    psr.add_argument("--run-name", type=str, required=True)

    # debugging
    psr.add_argument("--max-val-batches", default=-1, type=int)
    psr.add_argument("--show-metrics", type=str, nargs='+', default=["loss", "acc"])

    return psr.parse_args()

def prettyprint(args):
    return "\n".join([f"{argname}: {argval}" for argname, argval in vars(args).items()])
