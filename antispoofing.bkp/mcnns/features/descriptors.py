# -*- coding: utf-8 -*-

from antispoofing.mcnns.utils import *

# required for BSIF
import cv2
import bsif


class RawImage(object):

    def __init__(self):
        pass

    def extraction(self, img):
        return np.squeeze(img).astype(np.uint8)


class BSIF(object):

    def __init__(self, filter_dimensions):
        self.filter_dimensions = filter_dimensions

    def extraction(self, img):
        bsif_im = np.zeros_like(np.squeeze(img).astype(np.uint8))
        bsif_im = bsif.extract(np.squeeze(img).astype(np.uint8), bsif_im, self.filter_dimensions)

        # convert and scale the image to uint8
        scale = bsif_im/(2**self.filter_dimensions[2])
        bsif_im = scale*255

        # pdb.set_trace()
        # cv2.imwrite('bsifim.png', bsif_im.astype(np.uint8))

        return bsif_im.astype(np.uint8)
