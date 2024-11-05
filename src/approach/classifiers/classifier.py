import time
from copy import deepcopy
from collections import Counter
import torch
import numpy as np
from argparse import ArgumentParser
import umap
import pandas as pd

from loggers.exp_logger import ExperimentLogger
from datasets.exemplars_dataset import ExemplarsDataset
from abc import ABC, abstractmethod


class Classifier:
    """Basic class for implementing classifiers for incremental learning approaches"""

    def classify(self, task, outputs, features, targets, return_dists=False):
        pass

    def prototypes_update(self, t, trn_loader, transform):
        pass
