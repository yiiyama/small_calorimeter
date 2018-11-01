from __future__ import print_function

import numpy as np
import tensorflow as tf

class Converter(object):
    def __init__(self):
        pass

    def init_convert(self):
        pass

    def convert(self, event):
        return {}

    def feature_mapping(self):
        return {}

    def print_example(self, example):
        print(example)
