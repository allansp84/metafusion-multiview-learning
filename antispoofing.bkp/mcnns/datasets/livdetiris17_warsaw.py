# -*- coding: utf-8 -*-

import time
import itertools
from glob import glob
from antispoofing.mcnns.datasets.dataset import Dataset
from antispoofing.mcnns.utils import *


class LivDetIris17_Warsaw(Dataset):
    """ Interface for the LivDet Iris 2017 Warsaw dataset whose evaluation protocol is already defined by the orginizers.

    This class uses three auxiliary .text files provided by the LivDetIris2017's organizers, in which is defined the sets for training,
    testing and the unknown test.

    """

    def __init__(self, dataset_path, ground_truth_path='', permutation_path='', iris_location='',
                 output_path='./working', file_types=('.png', '.bmp', '.jpg', '.tiff'),
                 operation='crop', max_axis=320,
                 ):

        super(LivDetIris17_Warsaw, self).__init__(dataset_path, output_path, iris_location, file_types, operation, max_axis)

        self.sets = {}

        self.verbose = True

    def saving_ground_truth(self, all_fnames, all_labels, indexes, output_fname):

        labels = all_labels[indexes]

        fnames = []
        for fname in all_fnames[indexes]:
            fnames += [os.path.basename(fname)]

        dt = dict(names=('fnames', 'labels'), formats=('U300', np.int))

        aux = zip(fnames, labels)
        data = np.array(list(aux), dtype=dt)

        np.savetxt(output_fname, data, fmt="%s %d", delimiter=' ')

    def build_meta(self, inpath, filetypes):

        img_idx = 0

        all_fnames = []
        all_labels = []
        all_idxs = []
        train_idxs = []
        test_idxs = []
        val_idxs = []
        unknown_test_idxs = []

        hash_img_id = {}

        folders = [self.list_dirs(inpath, filetypes)]
        folders = sorted(list(itertools.chain.from_iterable(folders)))

        for i, folder in enumerate(folders):
            progressbar('-- folders', i, len(folders), new_line=True)

            fnames = [glob(os.path.join(inpath, folder, '*' + filetype)) for filetype in filetypes]
            fnames = sorted(list(itertools.chain.from_iterable(fnames)))

            for j, fname in enumerate(fnames):

                rel_path = os.path.relpath(fname, inpath)
                img_id, ext = os.path.splitext(os.path.basename(rel_path))

                if 'train/' in rel_path:

                    hash_img_id[img_id] = img_idx
                    train_idxs += [img_idx]
                    all_labels += [int('PRNT' in img_id)]
                    all_fnames += [fname]
                    all_idxs += [img_idx]
                    img_idx += 1

                elif 'test-known/' in rel_path:

                    hash_img_id[img_id] = img_idx
                    test_idxs += [img_idx]
                    all_labels += [int('PRNT' in img_id)]
                    all_fnames += [fname]
                    all_idxs += [img_idx]
                    img_idx += 1

                elif 'test-unknown/' in rel_path:

                    hash_img_id[img_id] = img_idx
                    unknown_test_idxs += [img_idx]
                    all_labels += [int('PRNT' in img_id)]
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
        unknown_test_idxs = np.array(unknown_test_idxs)

        # -- check if the training and testing sets are disjoint.
        try:
            assert not np.intersect1d(all_fnames[train_idxs], all_fnames[test_idxs]).size
            assert not np.intersect1d(all_fnames[train_idxs], all_fnames[unknown_test_idxs]).size
        except AssertionError:
            raise Exception('The training and testing sets are mixed')

        # # -- here we split train into train and validation
        # train_idxs_tmp = np.array(train_idxs)
        # s = train_idxs_tmp.shape[0]
        # np.random.seed(7)
        # tix = np.random.choice(s, int(s*0.8), replace=False)
        # vix = np.array(list(set(np.arange(s))-set(tix)))
        # train_idxs = train_idxs_tmp[tix]
        # val_idxs = train_idxs_tmp[vix]

        # self.saving_ground_truth(all_fnames, all_labels, train_idxs, output_fname='livdet-warsaw-train.txt')
        # self.saving_ground_truth(all_fnames, all_labels, test_idxs, output_fname='livdet-warsaw-test.txt')
        # self.saving_ground_truth(all_fnames, all_labels, unknown_test_idxs, output_fname='livdet-warsaw-unknown_test.txt')

        all_pos_idxs = np.where(all_labels[all_idxs] == self.POS_LABEL)[0]
        all_neg_idxs = np.where(all_labels[all_idxs] == self.NEG_LABEL)[0]

        r_dict = {'all_fnames': all_fnames,
                  'all_labels': all_labels,
                  'all_idxs': all_idxs,
                  'all_pos_idxs': all_pos_idxs,
                  'all_neg_idxs': all_neg_idxs,
                  'train_idxs': train_idxs,
                  'val_idxs': val_idxs,
                  'test_idxs': {'test': test_idxs,
                                'unknown_test': unknown_test_idxs,
                                'overall_test': np.concatenate((test_idxs, unknown_test_idxs)),
                                },

                  'hash_img_id': hash_img_id,
                  }

        if self.verbose:
            self.info(r_dict)

        return r_dict

    def protocol_eval(self, fold=0, n_fold=5, test_size=0.5, output_path=''):
        """ This method implement validation evaluation protocol for this dataset.

        Args:
            fold (int): This parameter is not used since this dataset already has the predefined subsets.
            n_fold (int): This parameter is not used since this dataset already has the predefined subsets.
            test_size (float): This parameter is not used since this dataset already has the predefined subsets.

        Returns:
            dict: A dictionary containing the training and testing sets.

        """

        # -- loading the training data and its labels
        all_fnames = self.meta_info['all_fnames']
        all_labels = self.meta_info['all_labels']
        train_idxs = self.meta_info['train_idxs']
        test_idxs = self.meta_info['test_idxs']

        all_data = self.get_imgs(all_fnames, sequenceid_col=0)

        # # -- create a mosaic for the positive and negative images.
        # print('-- all_fnames', all_fnames)
        # all_pos_idxs = np.where(all_labels[test_idxs['unknown_test']] == 1)[0]
        # all_neg_idxs = np.where(all_labels[test_idxs['unknown_test']] == 0)[0]
        # create_mosaic(all_data[all_pos_idxs], n_col=10, output_fname=os.path.join(self.output_path, 'mosaic-unknown_test-pos.{}.jpeg'.format(time.time())))
        # create_mosaic(all_data[all_neg_idxs], n_col=10, output_fname=os.path.join(self.output_path, 'mosaic-unknown_test-neg.{}.jpeg'.format(time.time())))

        train_set = {'data': all_data[train_idxs], 'labels': all_labels[train_idxs], 'idxs': train_idxs}

        test_set = {}
        for test_id in test_idxs:
            if test_idxs[test_id].size:
                test_set[test_id] = {'data': all_data[test_idxs[test_id]],
                                     'labels': all_labels[test_idxs[test_id]],
                                     'idxs': test_idxs[test_id],
                                     }

        # self.sets = {'train_set': train_set, 'test_set': test_set}
        return {'train_set': train_set, 'test_set': test_set}
