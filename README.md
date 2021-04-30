# Mixing Up Domain Generalization: Domain Metadata-Based Task Design for Robust Domain Representations

CS231N Project, Spring 2021, by `{tchang97, zhangbin}@stanford.edu`. Internal repo.

## Setup

The following was tested on Ubuntu 18.04 LTS on WSL2.

It is recommended that you create a virtual environment, i.e. `conda create -n WILDS python=3.7` for this.

1. Download the Google Cloud SDK following [these instructions](https://cloud.google.com/sdk/docs/install).
2. Download the `gcsfuse` utility for mounting GCloud buckets using [these instructions](https://github.com/GoogleCloudPlatform/gcsfuse/).
3. Initialize Google Cloud credentials by calling `gcloud init` and following the prompts. Make sure to set the current project to `CS231-DA-WILDS`.
4. Allow access to your Google Cloud resources (i.e. the storage bucket) by calling `gcloud auth login`; if this command hangs (e.g. the command line does not have direct access to your browser, as in WSL), try `gcloud auth login --no-launch-browser` instead.
5. Use `gcsfuse` to mount the storage bucket by calling `gcsfuse --implicit-dir cs231-wilds WILDS/`.
6. You can verify that the mount was successful via `ls WILDS`; you should see two folders titled `camelyon17_v1.0` and `iwildcam_v2.0`.
7. Run `pip install -r requirements.txt` to get the requisite packages.
8. Run `git submodule update --init --recursive` to clone into the submodules, used for baseline models.

## Incorporating DANN into baseline models.

1. I've included a DANN implementation as a submodule; most of it is likely not that useful. The only useful bit is the `ReverseLayer` implementation at `dann/function.py`; we'll do the split between the conv and fc layers of whatever model family is used for iWildCam and Camelyon (ResNet, DenseNet). The reversal layer only needs to be applied before the domain extraction architecture; i.e. dynamically in the `forward()` method.
