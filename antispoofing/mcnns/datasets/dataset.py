# -*- coding: utf-8 -*-

from abc import ABCMeta
from abc import abstractmethod
from antispoofing.mcnns.utils import *
import cv2


class Dataset(metaclass=ABCMeta):
    """ Abstract base class for the dataset that will be used.

    This abstract base class implements the cropping operation of the images that will be used in this work since this operation is
    independent of the dataset in use. Also this base class defines two methods that must me implemented by the subclasses, that is, the
    build_meta and protocol_eval methods. The build_meta contains the code responsible for creating the metadata of the dataset such as
    filenames, labels, indexes and some hash structures for all subsets contained into thet dataset. The protocol_eval method implements
    the evaluation protocol of the dataset.
    """

    def __init__(self, dataset_path, output_path, iris_location, file_types, operation, max_axis):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.file_types = file_types

        self.iris_location = iris_location
        self.operation = operation
        self.max_axis = max_axis

        # -- classes
        self.POS_LABEL = 1
        self.NEG_LABEL = 0

        self.__meta_info = None
        self.meta_was_built = False

        # temp directory for cache data
        self.hdf5_tmp_path = '/work/allansp/mcnns_tmp'

    def prune_train_dataset(self, all_labels, train_idxs):
        """ This method is responsible for pruning an input subset in order to get a balanced dataset in terms of their classes.

        Args:
            all_labels (numpy.ndarray):
            train_idxs (numpy.ndarray):

        Returns:

        """

        # -- prune samples if necessary to have equal sized splits
        neg_idxs = [idx for idx in train_idxs if all_labels[idx] == self.NEG_LABEL]
        pos_idxs = [idx for idx in train_idxs if all_labels[idx] == self.POS_LABEL]
        n_samples = min(len(neg_idxs), len(pos_idxs))

        rstate = np.random.RandomState(7)
        rand_idxs_neg = rstate.permutation(neg_idxs)
        rand_idxs_pos = rstate.permutation(pos_idxs)

        neg_idxs = rand_idxs_neg[:n_samples]
        pos_idxs = rand_idxs_pos[:n_samples]
        train_idxs = np.concatenate((pos_idxs, neg_idxs))

        return train_idxs

    @staticmethod
    def __crop_img(img, cx, cy, max_axis, padding=0):
        """ This method is responsible for cropping tha input image.

        Args:
            img (numpy.ndarray):
            cx (float):
            cy (float):
            max_axis (int):
            padding (int):

        Returns:
            numpy.ndarray: Cropped image.

        """

        new_height = max_axis
        new_width = max_axis

        cy -= new_height // 2
        cx -= new_width // 2

        if (cy + new_height) > img.shape[0]:
            shift = (cy + new_height) - img.shape[0]
            cy -= shift

        if (cx + new_width) > img.shape[1]:
            shift = (cx + new_width) - img.shape[1]
            cx -= shift

        cy = max(0., cy)
        cx = max(0., cx)

        cx = padding if cx == 0 else cx
        cy = padding if cy == 0 else cy

        cropped_img = img[cy - padding:cy + new_height + padding, cx - padding:cx + new_width + padding, :]

        return cropped_img

    def __get_iris_region(self, fnames, sequenceid_col):
        """

        Args:
            fnames (numpy.ndarray):

        Returns:

        """

        iris_location_list, iris_location_hash = read_csv_file(self.iris_location, sequenceid_col=sequenceid_col)

        segmentation_data = (iris_location_list, iris_location_hash)

        # origin_imgs, iris_location_list, iris_location_hash = load_images_hdf5(fnames,
        #                                                                        tmppath=self.hdf5_tmp_path,
        #                                                                        dsname=type(self).__name__.lower(),
        #                                                                        segmentation_data=segmentation_data,
        #                                                                        )

        npimg = load_images([fnames[0]])[0]

        imgs = []
        for i, fname in enumerate(fnames):

            origin_img = load_images([fname])[0]

            # -- try get the index for iris annotation using two different key (that's because we need to deal with different datasets)
            key_found = False
            k, index = 0, 0
            keys = [os.path.splitext(os.path.basename(fname))[0], os.path.splitext(os.path.relpath(fname, self.dataset_path))[0]]

            while not key_found and k < len(keys):
                try:
                    index = iris_location_hash[keys[k]]
                    key_found = True
                except KeyError:
                    k += 1
                    pass

            # -- If the index was found, we need to get the iris annotations in the list. Otherwise, an
            # -- exception (KeyError or IndexError) will be raised
            try:
                if key_found:
                    cx, cy = iris_location_list[index][-3:-1]
                else:
                    raise KeyError

            except KeyError:
                print('KeyError: Iris location not found. Cropping in the center of the image', fname)
                cy, cx = origin_img.shape[0] // 2, origin_img.shape[1] // 2

            except IndexError:
                print('IndexError: Iris location not found. Cropping in the center of the image', fname)
                cy, cx = origin_img.shape[0] // 2, origin_img.shape[1] // 2

            except:
                raise Exception('Error: Dataset.__get_iris_region')

            cx = int(float(cx))
            cy = int(float(cy))

            # -- If the cropped image do not have the minimum size required (self.max_axis), we resize
            # -- the image to attend this requirement. This operation preserves the aspect ratio of the image.

            # -- DEPRECATED
            # if min(origin_img.shape[:2]) < self.max_axis:
            #     ratio = self.max_axis/np.min(origin_img.shape[:2])
            #     new_shape = (int(origin_img.shape[1] * ratio), int(origin_img.shape[0] * ratio))
            #     origin_img = cv2.resize(origin_img, new_shape)
            #     if len(origin_img.shape) == 2:
            #         origin_img = origin_img[:, :, np.newaxis]

            if origin_img.shape[0] != npimg.shape[0] or origin_img.shape[1] != npimg.shape[1]:
                print("-- Resizing image", fname, "with shape", origin_img.shape)
                # resize the image to fit the current size, and update segmentation info
                origin_img, _, _ = resizeProp(origin_img, npimg.shape[:2], fname, segmentation_data=segmentation_data)

            img = self.__crop_img(origin_img, cx, cy, self.max_axis)

            imgs += [img]

        imgs = np.array(imgs, dtype=np.uint8)

        return imgs

    def get_imgs(self, input_fnames, sequenceid_col):

        fnames = input_fnames

        if 'crop' in self.operation:
            imgs = self.__get_iris_region(fnames, sequenceid_col)
        elif 'segment' in self.operation:
            imgs = self.__get_iris_region(fnames, sequenceid_col)
        else:
            imgs = load_images(fnames)

        return imgs

    @staticmethod
    def __check(labels_hash, liv_det_train_hash, liv_det_train_data):
        res = []
        base_names = []
        lenses_type = []
        base_names_liv_det = []
        for key in liv_det_train_hash.keys():
            found = 1
            try:
                labels_hash[key]
            except KeyError:
                print('Key not found in labels_hash[key]', key)
                sys.stdout.flush()
                found = 0

            try:
                liv_det_train_data[liv_det_train_hash[key]]
            except KeyError:
                print('Key not found in liv_det_train_data[liv_det_train_hash[key]]', key)
                sys.stdout.flush()
                found = 0

            if found:
                res += [int(liv_det_train_data[liv_det_train_hash[key]][1]) == int(labels_hash[key][1])]
                base_names += [labels_hash[key][0]]
                base_names_liv_det += [liv_det_train_data[liv_det_train_hash[key]][0]]
                lenses_type += [labels_hash[key][2]]

        res = np.array(res).reshape((-1, 1))
        base_names = np.array(base_names).reshape((-1, 1))
        base_names_liv_det = np.array(base_names_liv_det).reshape((-1, 1))
        lenses_type = np.array(lenses_type).reshape((-1, 1))

        return res, base_names, base_names_liv_det, lenses_type

    def info(self, meta_info):

        try:
            print('-*- Dataset Info -*-')
            print('-- all_labels:', meta_info['all_labels'].shape)
            print('-- train_idxs:', meta_info['train_idxs'].shape)
            print('   - pos:', np.where(meta_info['all_labels'][meta_info['train_idxs']] == self.POS_LABEL)[0].shape)
            print('   - neg:', np.where(meta_info['all_labels'][meta_info['train_idxs']] == self.NEG_LABEL)[0].shape)

            test_idxs = meta_info['test_idxs']
            for subset in test_idxs:
                print('-- %s:'%subset, test_idxs[subset].shape)
                print('   - pos:', np.where(meta_info['all_labels'][test_idxs[subset]] == self.POS_LABEL)[0].shape)
                print('   - neg:', np.where(meta_info['all_labels'][test_idxs[subset]] == self.NEG_LABEL)[0].shape)

            print('')
            sys.stdout.flush()
        except IndexError:
            pass

    @staticmethod
    def list_dirs(roo_tpath, file_types):
        """ This method returns the name of the subfolders that contain, at least, one file whose type is into the list file_types.

        Args:
            roo_tpath (str):
            file_types (Tuple[str]):

        Returns:
            list: Subfolders that contains at least one file of interest.

        """

        folders = []

        for root, dirs, files in os.walk(roo_tpath, followlinks=True):
            for f in files:
                if os.path.splitext(f)[1].lower() in file_types:
                    folders += [os.path.relpath(root, roo_tpath)]
                    break

        return folders

    def meta_info_feats(self, output_path, file_types):
        """ Metadata of the feature vectors.

        Args:
            output_path:
            file_types:

        Returns:
            dict: A dictionary contaning the metadata.
        """
        return self.build_meta(output_path, file_types)

    @property
    def meta_info(self):
        """ Metadata of the images.

        Returns:
            dict: A dictionary contaning the metadata.
        """

        if not self.meta_was_built:
            self.__meta_info = self.build_meta(self.dataset_path, self.file_types)
            self.meta_was_built = True

        return self.__meta_info

    @abstractmethod
    def build_meta(self, in_path, file_types):
        pass

    @abstractmethod
    def protocol_eval(self, fold=0, n_fold=5, test_size=0.5, output_path=''):
        pass
