# -*- coding: utf-8 -*-

import os
import sys
import itertools
import numpy as np
from glob import glob
from antispoofing.mcnns.datasets.dataset import Dataset
from antispoofing.mcnns.utils import *
import pdb


class NDContactLenses(Dataset):

    def __init__(self, dataset_path, ground_truth_path='', permutation_path='', iris_location='',
                 output_path='./working', file_types=('.png', '.bmp', '.jpg', '.tiff'),
                 operation='crop', max_axis=320,
                 ):

        super(NDContactLenses, self).__init__(dataset_path, output_path, iris_location, file_types, operation, max_axis)

        self.ground_truth_path = ground_truth_path
        self.permutation_path = permutation_path
        self.complete_gt = ('full_dataset' in ground_truth_path)
        self.verbose = True

    def read_permutation_file(self):

        data = np.genfromtxt(self.permutation_path, dtype=np.str, delimiter=',', skip_header=1)

        permuts_index = data[:, 0].astype(np.int)
        permuts_partition = data[:, 1]

        return permuts_index, permuts_partition

    # @profile
    def build_meta(self, inpath, filetypes):

        img_idx = 0

        all_fnames = []
        all_labels = []
        all_idxs = []
        train_idxs = []
        test_idxs = []
        unknown_test_idxs = []

        transparent_idxs = []
        texture_idxs = []
        no_contact_lenses_idxs = []
        hash_sensor = []
        sequence_id_train = []
        sequence_id_test = []
        hash_img_id = {}

        # -- for debugging
        cosmetic_idxs = []
        toric_idxs = []
        contact_lenses_idxs = []
        liv_det_labels = []
        labels_hash = {}
        sequence_id_transparent = []
        sequence_id_texture = []
        sequence_id_cosmetic = []
        sequence_id_toric = []
        sequence_id_no_contact_lenses = []
        sequence_id_contact_lenses = []

        folders = [self.list_dirs(inpath, filetypes)]
        folders = itertools.chain.from_iterable(folders)
        folders = sorted(list(folders))

        permuts_index, permuts_partition = self.read_permutation_file()

        if not self.complete_gt:
            # -- from the permutation files

            SEQUENCE_ID_COL = 0
            SENSOR_ID_COL = 3
            CONTACTS_COL = 4

            gt_list, gt_hash = read_csv_file(self.ground_truth_path, sequenceid_col=SEQUENCE_ID_COL)

            without_contact_lenses_idxs = np.where(gt_list[:, CONTACTS_COL] == 'N')[0].reshape(1, -1)

            with_contact_texture_idxs = np.where(gt_list[:, CONTACTS_COL] == 'T')[0].reshape(1, -1)
            with_contact_lenses = with_contact_texture_idxs

            idxs = np.where(permuts_partition == '0.8')[0]
            sequence_id_train = gt_list[permuts_index[idxs], SEQUENCE_ID_COL]

            idxs = np.where(permuts_partition == '0.2')[0]
            sequence_id_test = gt_list[permuts_index[idxs], SEQUENCE_ID_COL]

        else:
            # -- from the full_dataset.csv file

            SEQUENCE_ID_COL = 3
            SENSOR_ID_COL = 36
            CONTACTS_COL = 50                  # -- 'Yes',  'No',   'NULL', 'Unknown'
            CONTACTS_TYPE_COL = 51             # -- 'Soft', 'Hard', 'NULL', 'Unknown'
            CONTACTS_TEXTURE_COL = 52          # -- 'Yes',  'No',   'NULL', 'Unknown'
            CONTACTS_TORIC_COL = 53            # -- 'Yes',  'No',   'NULL', 'Unknown'
            CONTACTS_COSMETIC_COL = 54         # -- 'Yes',  'No',   'NULL', 'Unknown'
            TAG_LIST_COL = 61                  # -- manufacturer's name

            gt_list, gt_hash = read_csv_file(self.ground_truth_path, sequenceid_col=SEQUENCE_ID_COL)

            # -- for debugging
            with_contact_transparent_idxs = np.where(gt_list[:, CONTACTS_TYPE_COL] == 'Soft')[0]
            sequence_id_transparent = gt_list[with_contact_transparent_idxs, SEQUENCE_ID_COL]

            with_contact_texture_idxs = np.where(gt_list[:, CONTACTS_TEXTURE_COL] == 'Yes')[0]
            sequence_id_texture = gt_list[with_contact_texture_idxs, SEQUENCE_ID_COL]

            with_contact_cosmetic_idxs = np.where(gt_list[:, CONTACTS_COSMETIC_COL] == 'Yes')[0]
            sequence_id_cosmetic = gt_list[with_contact_cosmetic_idxs, SEQUENCE_ID_COL]

            with_contact_toric_idxs = np.where(gt_list[:, CONTACTS_TORIC_COL] == 'Yes')[0]
            sequence_id_toric = gt_list[with_contact_toric_idxs, SEQUENCE_ID_COL]

            without_contact_lenses_idxs = np.where(gt_list[:, CONTACTS_COL] == 'No')[0]
            sequence_id_no_contact_lenses = gt_list[without_contact_lenses_idxs, SEQUENCE_ID_COL]

            with_contact_lenses = np.where(gt_list[:, CONTACTS_COL] == 'Yes')[0]
            sequence_id_contact_lenses = gt_list[with_contact_lenses, SEQUENCE_ID_COL]

            sequence_id_tag_list = gt_list[:, TAG_LIST_COL]

            # rand_state = np.random.RandomState(7)
            # rand_seq_id_no_lenses = rand_state.permutation(sequence_id_no_contact_lenses)
            # rand_seq_id_lenses = rand_state.permutation(sequence_id_contact_lenses)
            # n_without_lenses = len(rand_seq_id_no_lenses)
            # n_with_lenses = len(rand_seq_id_lenses)
            # sequence_id_train = np.concatenate((rand_seq_id_no_lenses[:n_without_lenses//2], rand_seq_id_lenses[:n_with_lenses//2]))
            # sequence_id_test = np.concatenate((rand_seq_id_no_lenses[n_without_lenses//2:], rand_seq_id_lenses[n_with_lenses//2:]))

        sequence_id_pos_class = gt_list[with_contact_lenses, SEQUENCE_ID_COL]
        sequence_id_neg_class = gt_list[without_contact_lenses_idxs, SEQUENCE_ID_COL]

        # if self.verbose:
        #     print('with_contact_texture_idxs', with_contact_texture_idxs.shape)
        #     print('without_contact_lenses_idxs', without_contact_lenses_idxs.shape)
        #     print('sequence_id_pos_class', sequence_id_pos_class.shape)
        #     print('sequence_id_neg_class', sequence_id_neg_class.shape)
        #     sys.stdout.flush()

        liv_det_train_data, liv_det_train_hash = read_csv_file('extra/LivDet-Iris-2017_splits/livdet-train.txt',
                                                               sequenceid_col=0, delimiter=' ')
        liv_det_test_data, liv_det_test_hash = read_csv_file('extra/LivDet-Iris-2017_splits/livdet-test.txt',
                                                             sequenceid_col=0, delimiter=' ')
        liv_det_unknown_test_data, liv_det_unknown_test_hash = read_csv_file('extra/LivDet-Iris-2017_splits/livdet-unknown_test.txt',
                                                                             sequenceid_col=0, delimiter=' ')

        img_without_labels = 0
        for i, folder in enumerate(folders):
            progressbar('-- folders', i, len(folders), new_line=True)

            fnames = [glob(os.path.join(inpath, folder, '*' + filetype)) for filetype in filetypes]
            fnames = itertools.chain.from_iterable(fnames)
            fnames = sorted(list(fnames))

            for j, fname in enumerate(fnames):

                rel_path = os.path.relpath(fname, inpath)
                img_id, ext = os.path.splitext(os.path.basename(rel_path))
                img_id = img_id.split('_')[0]

                line = ['a']
                # try:
                #     line = gt_hash[img_id]
                # except KeyError:
                #     pass

                # -- check if the current image is in the ground truth file
                if line:

                    # if img_id in sequence_id_train:
                    if img_id in liv_det_train_hash:

                        if not (img_id in hash_img_id):
                            hash_img_id[img_id] = img_idx
                            train_idxs += [img_idx]
                            all_labels += [int(liv_det_train_data[liv_det_train_hash[img_id]][1])]
                            all_fnames += [fname]
                            all_idxs += [img_idx]
                            img_idx += 1
                            # hash_sensor += [gt_list[gt_hash[img_id]][SENSOR_ID_COL]]
                            # all_labels += [self.POS_LABEL if img_id in sequence_id_pos_class else self.NEG_LABEL]

                    elif img_id in liv_det_test_hash:

                        if not (img_id in hash_img_id):
                            hash_img_id[img_id] = img_idx
                            test_idxs += [img_idx]
                            all_labels += [int(liv_det_test_data[liv_det_test_hash[img_id]][1])]
                            all_fnames += [fname]
                            all_idxs += [img_idx]
                            img_idx += 1

                    # elif img_id in sequence_id_test:
                    elif img_id in liv_det_unknown_test_hash:

                        if not (img_id in hash_img_id):
                            hash_img_id[img_id] = img_idx
                            unknown_test_idxs += [img_idx]
                            all_labels += [int(liv_det_unknown_test_data[liv_det_unknown_test_hash[img_id]][1])]
                            all_fnames += [fname]
                            all_idxs += [img_idx]
                            img_idx += 1
                            # hash_sensor += [gt_list[gt_hash[img_id]][SENSOR_ID_COL]]
                            # all_labels += [self.POS_LABEL if img_id in sequence_id_pos_class else self.NEG_LABEL]

                    else:
                        # print('Image not found in the dataset!')
                        # sys.stdout.flush()
                        pass

                    # -- for debugging
                    # if img_id in sequence_id_no_contact_lenses:
                    #     no_contact_lenses_idxs += [img_idx]
                    #     all_fnames += [fname]
                    #     all_idxs += [img_idx]
                    #     all_labels += [0]
                    #     img_idx += 1
                    #
                    #     try:
                    #         labels_hash[img_id] += [0]
                    #     except KeyError:
                    #         labels_hash[img_id] = [0]
                    #
                    # if img_id in sequence_id_texture:
                    #     texture_idxs += [img_idx]
                    #     all_fnames += [fname]
                    #     all_idxs += [img_idx]
                    #     all_labels += [1]
                    #     img_idx += 1
                    #
                    #     try:
                    #         labels_hash[img_id] += [1]
                    #     except KeyError:
                    #         labels_hash[img_id] = [1]
                    #
                    # if img_id in sequence_id_cosmetic:
                    #     cosmetic_idxs += [img_idx]
                    #     all_fnames += [fname]
                    #     all_idxs += [img_idx]
                    #     all_labels += [1]
                    #     img_idx += 1
                    #
                    #     try:
                    #         labels_hash[img_id] += [1]
                    #     except KeyError:
                    #         labels_hash[img_id] = [1]
                    #
                    # if img_id in sequence_id_transparent:
                    #     transparent_idxs += [img_idx]
                    #     all_fnames += [fname]
                    #     all_idxs += [img_idx]
                    #     all_labels += [1]
                    #     img_idx += 1
                    #
                    #     try:
                    #         labels_hash[img_id] += [1]
                    #     except KeyError:
                    #         labels_hash[img_id] = [1]
                    #
                    # if img_id in sequence_id_toric:
                    #     toric_idxs += [img_idx]
                    #     all_fnames += [fname]
                    #     all_idxs += [img_idx]
                    #     all_labels += [1]
                    #     img_idx += 1
                    #
                    #     try:
                    #         labels_hash[img_id] += [1]
                    #     except KeyError:
                    #         labels_hash[img_id] = [1]
                    #
                    # if img_id in sequence_id_contact_lenses:
                    #     contact_lenses_idxs += [img_idx]
                    #     all_fnames += [fname]
                    #     all_idxs += [img_idx]
                    #     all_labels += [1]
                    #     img_idx += 1
                    #
                    #     try:
                    #         labels_hash[img_id] += [1]
                    #     except KeyError:
                    #         labels_hash[img_id] = [1]
                    #
                    # else:
                    #     img_without_labels += 1
                    #     if not (img_id in labels_hash):
                    #         labels_hash[img_id] = [fname, 1, 'none']
                    #     else:
                    #         print("Image is already in the hash [else]", img_id)
                    #
                    # try:
                    #     liv_det_labels += [liv_det_train_data[liv_det_train_hash[img_id]]]
                    # except KeyError:
                    #     liv_det_labels += [9]

        all_fnames = np.array(all_fnames)
        all_labels = np.array(all_labels)
        all_idxs = np.array(all_idxs)
        train_idxs = np.array(train_idxs)
        test_idxs = np.array(test_idxs)
        unknown_test_idxs = np.array(unknown_test_idxs)

        # train_idxs = self.prune_train_dataset(all_labels, train_idxs)

        # -- for debugging
        # dt = dict(names=('fname', 'label'), formats=('U300', np.int))
        # aux = zip()
        # aux_ = np.array(list(aux), dtype=dt)
        # check_train, base_names, base_names_liv_det_train, lenses_type = self.check(labels_hash, liv_det_train_hash, liv_det_train_data)
        # idx = np.where(check_train == False)[0]
        # np.savetxt('base_names_train.txt', base_names[idx], fmt='%s')
        # np.savetxt('base_names_liv_det_train.txt', np.concatenate((base_names_liv_det_train[idx], lenses_type[idx]), axis=1), fmt='%s %s')
        #
        # check_test, base_names, base_names_liv_det_test, lenses_type = self.check(labels_hash, liv_det_test_hash, liv_det_test_data)
        # idx = np.where(check_test == False)[0]
        # np.savetxt('base_names_test.txt', base_names[idx], fmt='%s')
        # np.savetxt('base_names_liv_det_test.txt', np.concatenate((base_names_liv_det_test[idx], lenses_type[idx]), axis=1), fmt='%s %s')
        #
        # check_unknown_test, base_names, base_names_liv_det_unknown_test, lenses_type = self.check(labels_hash, liv_det_unknown_test_hash, liv_det_unknown_test_data)
        # idx = np.where(check_unknown_test == False)[0]
        # np.savetxt('base_names_unknown_test.txt', base_names[idx], fmt='%s')
        # np.savetxt('base_names_liv_det_unknown_test.txt', np.concatenate((base_names_liv_det_unknown_test[idx], lenses_type[idx]), axis=1), fmt='%s %s')

        # if self.complete_gt:
        #     pdb.set_trace()

        pdb.set_trace()

        all_pos_idxs = np.where(all_labels[all_idxs] == self.POS_LABEL)[0]
        all_neg_idxs = np.where(all_labels[all_idxs] == self.NEG_LABEL)[0]

        if self.verbose:
            print("-- all_fnames:", all_fnames.shape)
            print("-- all_labels:", all_labels.shape)
            print("-- all_idxs:", all_idxs.shape)
            print("-- train_idxs:", train_idxs.shape)
            print("   -- pos:", np.where(all_labels[train_idxs] == self.POS_LABEL)[0].shape)
            print("   -- neg:", np.where(all_labels[train_idxs] == self.NEG_LABEL)[0].shape)
            print("-- test_idxs:", test_idxs.shape)
            print("   -- pos:", np.where(all_labels[test_idxs] == self.POS_LABEL)[0].shape)
            print("   -- neg:", np.where(all_labels[test_idxs] == self.NEG_LABEL)[0].shape)
            print("-- unknown_test_idxs:", unknown_test_idxs.shape)
            print("   -- pos:", np.where(all_labels[unknown_test_idxs] == self.POS_LABEL)[0].shape)
            print("   -- neg:", np.where(all_labels[unknown_test_idxs] == self.NEG_LABEL)[0].shape)

            sys.stdout.flush()

        r_dict = {'all_fnames': all_fnames,
                  'all_labels': all_labels,
                  'all_idxs': all_idxs,

                  'train_idxs': train_idxs,
                  'test_idxs': test_idxs,
                  'unknown_test_idxs': unknown_test_idxs,

                  'all_pos_idxs': all_pos_idxs,
                  'all_neg_idxs': all_neg_idxs,

                  'texture_idxs': texture_idxs,
                  'transparent_idxs': transparent_idxs,
                  'no_contact_lenses_idxs': no_contact_lenses_idxs,

                  'hash_img_id': hash_img_id,
                  }

        return r_dict
