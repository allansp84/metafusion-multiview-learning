# -*- coding: utf-8 -*-

from antispoofing.mcnns.classification.cnn import CNN


ml_algo = {0: CNN,
           # 1: LinearSVM
           }


losses_functions = {0: 'categorical_crossentropy',
                    1: 'sparse_categorical_crossentropy',
                    2: 'categorical_hinge',
                    3: 'hinge',
                    4: 'binary_crossentropy',
                    }


optimizer_methods = {0: 'SGD',
                     1: 'Adam',
                     2: 'Adagrad',
                     3: 'Adadelta',
                     4: 'Adamax',
                     }
