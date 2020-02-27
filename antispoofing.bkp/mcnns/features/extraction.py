# -*- coding: utf-8 -*-

import os
import sys

from antispoofing.mcnns.features.descriptors import *
from antispoofing.mcnns.utils import *


class Extraction(object):

    def __init__(self, output_fnames, input_fnames,
                 descriptor='spoofnet-a',
                 file_type="png",
                 params='',
                 img=None
                 ):

        self.output_fnames = output_fnames
        self.input_fnames = input_fnames
        self.descriptor = descriptor
        self.params = params
        self.file_type = file_type

        self.debug = True

        self.flatten = True
        self.saveoutput = True

        # in the case an image is passed, extraction will operate on that image.
        # otherwise it will use the file names
        self.img = img
        if not self.img is None:
            self.output_fnames = []
            self.input_fnames = []
            self.saveoutput = False

    def extract_features(self, imgs):

        feature_descriptor = None

        if 'rawimage' in str(self.descriptor).lower():
            feature_descriptor = RawImage()
        elif 'bsif' in self.descriptor:
            fsize = np.array(eval(self.params), dtype=np.uint8)
            feature_descriptor = BSIF(filter_dimensions=fsize)
        else:
            pass

        feature_vector = []
        for idx in range(len(imgs)):
            feats = feature_descriptor.extraction(imgs[idx, :, :, :])
            feature_vector.append(feats)
        feature_vector = np.array(feature_vector)

        return feature_vector

    def save_features(self, output_fnames, feats):

        for feat, output_fname in zip(feats, output_fnames):

            if self.debug:
                print("-- saving {0} features extracted from {1}".format(self.descriptor, output_fname))
                sys.stdout.flush()

            try:
                os.makedirs(os.path.dirname(output_fname))
            except OSError:
                pass

            # feat = np.reshape(feat, (1, -1))
            # np.save(output_fname, feat)
            cv2.imwrite(output_fname, feat)

        return True

    def run(self):

        if not isinstance(self.input_fnames, list):
            self.input_fnames = np.array([self.input_fnames])

        if not isinstance(self.output_fnames, list):
            self.output_fnames = np.array([self.output_fnames])

        exists = []
        for idx in range(len(self.output_fnames)):
            exists += [os.path.exists(self.output_fnames[idx])]

        if not exists.count(False):
            for idx in range(len(self.output_fnames)):
                if self.debug:
                    print("-- done {0}".format(self.output_fnames[idx]), flush=True)
            return True

        imsize = (1, -1)
        if self.img is None:
            imgs = load_images(self.input_fnames,
                               #dsname=self.descriptor.lower(),
                              )
        else:
            imsize = self.img.shape
            imgs = np.array([self.img], dtype=np.uint8)

        feats = self.extract_features(imgs)

        if self.saveoutput:
            self.save_features(self.output_fnames, feats)
            return True

        # if not saving, we return the output as an image
        return np.reshape(feats, imsize)
