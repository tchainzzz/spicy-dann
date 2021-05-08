from argparse import ArgumentParser

def get_opts():
    psr = ArgumentParser()

    # data
    psr.add_argument("--dataset", required=True, type=str, choices=['camelyon17', 'iwildcam'])

    # training
    psr.add_argument("--model-name", type=str, required=True)
    psr.add_argument("--batch-size", default=16, type=int)
    psr.add_argument("--n-epochs", default=50, type=int)
    psr.add_argument("--get-train-metrics", action='store_true')

    # checkpointing
    psr.add_argument("--save-every", default=5, type=int)

    # debugging
    psr.add_argument("--max-val-batches", default=-1, type=int)

    return psr.parse_args()

def prettyprint(args):
    return "\n".join([f"{argname}: {argval}" for argname, argval in vars(args).items()])
