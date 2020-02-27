# -*- coding: utf-8 -*-

import os
import itertools
import numpy as np
from glob import glob
from antispoofing.mcnns.datasets.dataset import Dataset
from antispoofing.mcnns.utils import *


class ForAnnotation(Dataset):

    def __init__(self, dataset_path, ground_truth_path='', permutation_path='', iris_location='',
                 output_path='./working', file_types=('.png', '.bmp', '.jpg', '.tiff'),
                 operation='crop', max_axis=320,
                 ):

        super(ForAnnotation, self).__init__(dataset_path, output_path, iris_location, file_types, operation, max_axis)
        self.verbose = True

    def build_meta(self, inpath, filetypes):

        img_idx = 0

        all_fnames = []
        all_labels = []
        all_idxs = []
        train_idxs = []
        test_idxs = []

        hash_img_id = {}

        folders = [self.list_dirs(inpath, filetypes)]
        folders = sorted(list(itertools.chain.from_iterable(folders)))

        for i, folder in enumerate(folders):

            fnames = [glob(os.path.join(inpath, folder, '*' + filetype)) for filetype in filetypes]
            fnames = sorted(list(itertools.chain.from_iterable(fnames)))

            for j, fname in enumerate(fnames):

                rel_path = os.path.relpath(fname, inpath)
                img_id, ext = os.path.splitext(os.path.basename(rel_path))
                img_id = img_id.split('_')[0]

                if not (img_id in hash_img_id):
                    hash_img_id[img_id] = img_idx
                    test_idxs += [img_idx]

                    all_labels += [0]
                    all_fnames += [fname]
                    all_idxs += [img_idx]

                    img_idx += 1
                else:
                    pass

        all_fnames = np.array(all_fnames)
        all_labels = np.array(all_labels)
        all_idxs = np.array(all_idxs)
        train_idxs = np.array(train_idxs)
        test_idxs = np.array(test_idxs)

        r_dict = {'all_fnames': all_fnames,
                  'all_labels': all_labels,
                  'all_idxs': all_idxs,

                  'train_idxs': train_idxs,
                  'test_idxs': {'test': test_idxs,
                                },

                  'hash_img_id': hash_img_id,
                  }

        if self.verbose:
            self.info(r_dict)

        return r_dict

    # def _additional_test_set(self, key, test_set):
    #     """
    #
    #     Args:
    #         key:
    #         test_set:
    #
    #     Returns:
    #
    #     """
    #     self.sets['test_set'][key] = test_set

    def protocol_eval(self, fold=0, n_fold=5, test_size=0.5):
        """

        Args:
            fold:
            n_fold:
            test_size:

        Returns:

        """

        # -- loading the training data and its labels
        all_fnames = self.meta_info['all_fnames']
        all_labels = self.meta_info['all_labels']
        train_idxs = self.meta_info['train_idxs']
        test_idxs = self.meta_info['test_idxs']

        all_data = self.get_imgs(all_fnames)

        # # -- create a mosaic for the positive and negative images.
        # all_pos_idxs = np.where(all_labels == 1)[0]
        # all_neg_idxs = np.where(all_labels == 0)[0]
        # create_mosaic(all_data[all_pos_idxs][:20], n_col=10, output_fname=os.path.join(self.output_path, 'mosaic-pos-class-1.jpeg'))
        # create_mosaic(all_data[all_neg_idxs][:20], n_col=10, output_fname=os.path.join(self.output_path, 'mosaic-neg-class-0.jpeg'))

        train_set = {}

        test_set = {}
        for test_id in test_idxs:
            if test_idxs[test_id].size:
                test_set[test_id] = {'data': all_data[test_idxs[test_id]],
                                     'labels': all_labels[test_idxs[test_id]],
                                     'idxs': test_idxs[test_id],
                                     }

        # self.sets = {'train_set': train_set, 'test_set': test_set}
        return {'train_set': train_set, 'test_set': test_set}

    # def protocol_sets(self, fold=0, n_fold=5, test_size=0.5, dataset_b=None):
    #     """
    #
    #     Args:
    #         fold:
    #         n_fold:
    #         test_size:
    #         protocol:
    #         dataset_b:
    #
    #     Returns:
    #
    #     """
    #
    #     if dataset_b is None:
    #         # -- get the official training and testing sets of the first dataset
    #         self.protocol_eval(fold=fold, n_fold=n_fold, test_size=test_size)
    #     else:
    #
    #         # -- prepare the official training and testing sets of the first dataset
    #         self.protocol_eval(fold=fold, n_fold=n_fold, test_size=test_size)
    #
    #         # -- prepare the official training and testing sets of the second dataset
    #         dataset_b = dataset_b.protocol_eval(fold=fold, n_fold=n_fold, test_size=test_size)
    #
    #         # -- add the testing set of the second dataset in the first dataset
    #         self._additional_test_set('{0}_test'.format(str(dataset_b.__class__.__name__).lower()),
    #                                   test_set=dataset_b['test_set'],
    #                                   )
    #
    #     return self.sets
