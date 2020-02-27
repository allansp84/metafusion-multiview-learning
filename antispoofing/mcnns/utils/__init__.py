# -*- coding: utf-8 -*-

# -- common functions and constants
from antispoofing.mcnns.utils.constants import N_JOBS
from antispoofing.mcnns.utils.constants import CONST
from antispoofing.mcnns.utils.constants import PROJECT_PATH
from antispoofing.mcnns.utils.constants import UTILS_PATH
from antispoofing.mcnns.utils.constants import SEED


from antispoofing.mcnns.utils.misc import modification_date
from antispoofing.mcnns.utils.misc import get_time
from antispoofing.mcnns.utils.misc import total_time_elapsed
from antispoofing.mcnns.utils.misc import RunInParallel
from antispoofing.mcnns.utils.misc import progressbar
from antispoofing.mcnns.utils.misc import save_object
from antispoofing.mcnns.utils.misc import load_object
from antispoofing.mcnns.utils.misc import load_images
from antispoofing.mcnns.utils.misc import load_images_hdf5
from antispoofing.mcnns.utils.misc import resizeProp
from antispoofing.mcnns.utils.misc import mosaic
from antispoofing.mcnns.utils.misc import read_csv_file
from antispoofing.mcnns.utils.misc import get_interesting_samples
from antispoofing.mcnns.utils.misc import create_mosaic
from antispoofing.mcnns.utils.misc import retrieve_fnames_by_type
from antispoofing.mcnns.utils.misc import classification_results_summary
from antispoofing.mcnns.utils.misc import get_imgs
from antispoofing.mcnns.utils.misc import convert_numpy_dict_items_to_list
from antispoofing.mcnns.utils.misc import unique_everseen


# -- common imports
import pdb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import mlab

import os
import sys
import time

# os.environ['PYTHONHASHSEED'] = '0'

import numpy as np
# np.random.rand(SEED)

# import tensorflow as tf
# # tf.set_random_seed(SEED)

# import keras
# from keras import activations
# from keras import backend as K
# from keras.models import Sequential
# from keras.layers import Input
# from keras.layers import InputLayer
# from keras.layers import Conv2D
# from keras.layers import Activation
# from keras.layers import LocallyConnected2D
# from keras.layers import advanced_activations
# from keras.layers.core import Flatten
# from keras.layers.core import Dense
# from keras.layers.core import Layer
# from keras.layers.convolutional import MaxPooling2D
# from keras.layers.normalization import BatchNormalization
# from keras.preprocessing.image import ImageDataGenerator
