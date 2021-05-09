def dict_formatter(d, delimiter=":", joiner="-"):
    return f" {joiner} ".join([f"{k}{delimiter} {v:.4f}" for k, v in d.items()])