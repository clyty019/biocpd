#!/usr/bin/env python
# coding: utf-8
"""Common utility functions"""

import random
import numpy as np


def set_seed(seed=42):
    """Set the random seed for numpy and random to ensure result consistency"""
    random.seed(seed)
    np.random.seed(seed)
