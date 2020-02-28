import tensorflow as tf
import os
from model import Model
from dataset import *

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

model = Model()


if __name__ == "__main__":
    pass
