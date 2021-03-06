# -*- coding: utf-8 -*-

from hskl.demo import dl_hyrank, load_hyrank
import os

# Download, unpack, and load HyRANK dataset
path = os.getcwd()
dl_hyrank(path)
train, label, validation = load_hyrank(path)

# TODO: hskl-segmentation