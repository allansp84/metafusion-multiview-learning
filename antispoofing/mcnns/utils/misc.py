# -*- coding: utf-8 -*-

import sys
import numpy as np
import itertools as it
from multiprocessing import Pool, Value, Lock
from antispoofing.mcnns.utils.constants import *
import datetime
import pickle
from PIL import Image
import cv2
from skimage import measure
from operator import itemgetter
import pdb
import csv
from glob import glob
import tables
import time


counter = Value('i', 0)
counter_lock = Lock()


def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.datetime.fromtimestamp(t)


def get_time():
    return datetime.datetime.now()


def total_time_elapsed(start, finish):
    elapsed = finish - start

    total_seconds = int(elapsed.total_seconds())
    total_minutes = int(total_seconds // 60)
    hours = int(total_minutes // 60)
    minutes = int(total_minutes % 60)
    seconds = int(round(total_seconds % 60))

    return "{0:02d}+{1:02d}:{2:02d}:{3:02d} (total_seconds: {4})".format(elapsed.days, hours, minutes, seconds, elapsed.total_seconds())


def progressbar(name, current_task, total_task, bar_len=20, new_line=False):
        percent = float(current_task) / total_task

        progress = ""
        for i in range(bar_len):
            if i < int(bar_len * percent):
                progress += "="
            else:
                progress += " "

        print("\r{0}{1}: [{2}] {3}/{4} ({5:.1f}%).{6:30}".format(CONST.OK_GREEN, name,
                                                                 progress, current_task,
                                                                 total_task, percent * 100,
                                                                 CONST.END),
              end="")

        if new_line:
            print()

        sys.stdout.flush()


def start_process():
    pass


def launch_tasks(arg):

    global counter
    global counter_lock

    index, n_tasks, task = arg

    result = task.run()

    with counter_lock:
        # elapsed = datetime.datetime.now() - start_time
        # time_rate = ((counter.value-1) * previous_time_rate + elapsed.total_seconds())/float(counter.value)
        counter.value += 1
        progressbar('-- RunInParallel', counter.value, n_tasks)
        # print('\r{0}-- RunInParallel: {1} task(s) done.{2:30}'.format(CONST.OK_GREEN, counter.value, CONST.END), end="")
        # sys.stdout.flush()

    return result


class RunInParallel(object):
    def __init__(self, tasks, n_proc=N_JOBS):

        # -- private attributes
        self.__pool = Pool(initializer=start_process, processes=n_proc)
        self.__tasks = []

        # -- public attributes
        self.tasks = tasks

    @property
    def tasks(self):
        return self.__tasks

    @tasks.setter
    def tasks(self, tasks_list):
        self.__tasks = []
        for i, task in enumerate(tasks_list):
            self.__tasks.append((i, len(tasks_list), task))

    def run(self):
        global counter
        counter.value = 0

        pool_outs = self.__pool.map_async(launch_tasks, self.tasks)
        self.__pool.close()
        self.__pool.join()

        try:
            work_done = [out for out in pool_outs.get() if out]
            assert (len(work_done)) == len(self.tasks)

            print('\n{0}-- finish.{1:30}'.format(CONST.OK_GREEN, CONST.END))
            sys.stdout.flush()

        except AssertionError:
            print('\n{0}ERROR: some objects could not be processed!{1:30}\n'.format(CONST.ERROR, CONST.END))
            sys.exit(1)


def save_object(obj, fname):

    try:
        os.makedirs(os.path.dirname(fname))
    except OSError:
        pass

    fo = open(fname, 'wb')
    pickle.dump(obj, fo)
    fo.close()


def load_object(fname):
    fo = open(fname, 'rb')
    obj = pickle.load(fo)
    fo.close()

    return obj


def preprocessing(img):

    gimg = cv2.medianBlur(img.mean(axis=2).astype(np.uint8), 31)
    _, bimg = cv2.threshold(gimg, 10, 255, cv2.THRESH_BINARY_INV)

    size = 3
    kernel = np.ones((size, size), np.uint8)
    bimg = cv2.morphologyEx(bimg[size:-size, size:-size], cv2.MORPH_OPEN, kernel, iterations=3)
    bimg = np.pad(bimg, size, 'constant', constant_values=0)

    return bimg


def find_bounding_box(contour):
    min_x, max_x = contour[:, 0].min(), contour[:, 0].max()
    min_y, max_y = contour[:, 1].min(), contour[:, 1].max()
    width = max_x - min_x
    height = max_y - min_y
    return np.array([min_x, min_y, width, height])


def get_center_of_iris_image(img):

    bimage = preprocessing(img)

    labels = measure.label(bimage)

    label_number = 0

    results = []
    while True:
        temp = np.uint8(labels == label_number) * 255
        if not cv2.countNonZero(temp):
            break
        results.append(temp)
        label_number += 1
    results = np.array(results)

    db = []
    for res in results:
        images, contours, hierarchy = cv2.findContours(res.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):

            if contour.shape[0] > 2:

                # contour = cv2.convexHull(contour)

                # -- compute the circularity
                area = cv2.contourArea(contour)

                # -- finding the bounding box
                bbox = find_bounding_box(np.squeeze(contour))

                # -- compute the circularity
                circumference = cv2.arcLength(contour, True)
                circularity = circumference ** 2 / (4 * np.pi * area)

                approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
                n_approx = len(approx)

                aux_img = np.zeros(bimage.shape, dtype=np.uint8) + 255
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.array(box, np.int)
                cv2.drawContours(aux_img, [box], 0, (0,0,0), cv2.FILLED)
                min_rect_area = np.count_nonzero(255 - aux_img)

                aux_img = np.zeros(bimage.shape, dtype=np.uint8) + 255
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(aux_img, center, radius, (0, 0, 0), cv2.FILLED)
                min_enclosing_circle_area = np.count_nonzero(255 - aux_img)

                if not min_enclosing_circle_area:
                    min_enclosing_circle_area = 1.

                compactness = min_rect_area/min_enclosing_circle_area

                M = cv2.moments(contour)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                center_of_mass = [cx, cy]

                db.append([area, circularity, center_of_mass, bbox, n_approx, compactness])

    db = sorted(db, key=itemgetter(0), reverse=True)

    center_of_mass = []
    idx, found = 0, 0
    while idx < len(db) and not found:
        area, circularity, center_of_mass, bbox, n_approx, compactness = db[idx]

        x, y, width, height = bbox

        # print("area, circularity, center_of_mass, n_approx, compactness")
        # print(area, circularity, center_of_mass, n_approx, compactness)
        # sys.stdout.flush()

        if circularity > 0.9 and circularity < 1.2:
            if area > 30*30 and area < 104*104:
                if img[y:y+height, x:x+width, :].mean() < 50:
                    found = 1
        idx += 1

    return center_of_mass


def __crop_img(img, cx, cy, max_axis, padding=0):
    new_height = max_axis
    new_width = max_axis

    cy -= new_height//2
    cx -= new_width//2

    if (cy + new_height) > img.shape[0]:
        shift = (cy + new_height) - img.shape[0]
        cy -= shift

    if (cx + new_width) > img.shape[1]:
        shift = (cx + new_width) - img.shape[1]
        cx -= shift

    cy = max(0, cy)
    cx = max(0, cx)

    cx = padding if cx == 0 else cx
    cy = padding if cy == 0 else cy

    cropped_img = img[cy - padding:cy + new_height + padding, cx - padding:cx + new_width + padding, :]

    return cropped_img


def __resize_img(img, max_axis):
    ratio = max_axis / np.max(img.shape)
    n_rows, n_cols = img.shape[:2]
    new_n_rows = int(n_rows * ratio)
    new_n_cols = int(n_cols * ratio)

    new_shape = (new_n_rows, new_n_cols, img.shape[2])

    return np.resize(img, new_shape)


def __try_get_iris_region(img, max_axis):

    center_of_mass = get_center_of_iris_image(img)
    # center_of_mass = intensity_profile(img)

    if len(center_of_mass) == 0:
        center_of_mass = [img.shape[1]//2, img.shape[0]//2]

    cx, cy = center_of_mass
    img = __crop_img(img, cx, cy, max_axis)

    return img


def intensity_profile(img):
    """Intensity profile available in https://github.com/sbanerj1/IrisSeg

    Reference:
    S. Banerjee and D. Mery. Iris Segmentation using Geodesic Active Contours and GrabCut.
    In Workshop on 2D & 3D Geometric Properties from Incomplete Data at PSIVT (PSIVT Workshops), 2015.
    """

    h, w, d = img.shape
    h3 = h // 3
    w3 = w // 3

    lft = 1 * w3
    rt = 2 * w3
    up = 1 * h3
    down = 2 * h3

    hor_l = [0] * (int(down - up) // 5 + 1)
    ver_l = [0] * (int(rt - lft) // 5 + 1)
    temp_l = []
    hor_list = []
    ver_list = []
    min_val = 100
    ellipse_size = 0
    min_x = 0
    min_y = 0
    maxf = 0
    maxs = 0
    eoc = 0

    i = lft
    j = up
    while i <= rt:
        j = up
        while j <= down:
            if int(img[j][i][0]) < min_val:
                min_val = int(img[j][i][0])
            j += 1
        i += 1

    m = 0
    n = up
    k = 0
    max_blah = 0
    while n <= down:
        m = lft
        while m <= rt:
            temp = int(img[n][m][0])
            if temp < (min_val + 10):
                hor_l[k] += 1
                temp_l.append([m, n])
            else:
                pass
            m += 1
        if hor_l[k] > max_blah:
            max_blah = hor_l[k]
            hor_list = temp_l
        temp_l = []
        n += 5
        k += 1

    max_t = max_blah

    m = 0
    n = lft
    k = 0
    max_blah = 0
    temp_l = []
    while n <= rt:
        m = up
        while m <= down:
            temp = int(img[m][n][0])
            if temp < (min_val + 10):
                ver_l[k] += 1
                temp_l.append([n, m])
            else:
                pass
            m += 1
        if ver_l[k] > max_blah:
            max_blah = ver_l[k]
            ver_list = temp_l
        temp_l = []
        n += 5
        k += 1

    if max_blah > max_t:
        max_t = max_blah

    cx = 0
    cy = 0
    hlst = []
    vlst = []
    sumh = 0
    sumv = 0

    i = lft

    while i <= rt:
        j = up
        while j <= down:
            if int(img[j][i][0]) < (min_val + 10):
                hlst.append(i)
                sumh += i
                vlst.append(j)
                sumv += j
            j += 1
        i += 1

    cx = int(sumh // len(hlst))
    cy = int(sumv // len(vlst))

    return [cx, cy]


def resizeProp(img, dstshape, name, segmentation_data, max_axis=260):

    iris_data = segmentation_data[0]
    iris_map = segmentation_data[1]

    # calculate the resize factor, based on the average radius of the other images
    rfactor = max_axis / img.shape[1]
    if rfactor > 1:
        rfactor = 1

    rsimg = cv2.resize(img, (0, 0), fx=rfactor, fy=rfactor)

    # pad the image to the appropriate size reflecting the border rows/columns
    yoffset = (dstshape[0]-rsimg.shape[0])//2
    yoffset_ = abs(((yoffset * 2) + rsimg.shape[0]) - dstshape[0])
    xoffset = (dstshape[1]-rsimg.shape[1])//2
    xoffset_ = abs(((xoffset * 2) + rsimg.shape[1]) - dstshape[1])

    newimg = cv2.copyMakeBorder(rsimg, yoffset, yoffset+yoffset_, xoffset, xoffset+xoffset_, cv2.BORDER_REFLECT)
    newimg = newimg[:,:,np.newaxis]

    key = os.path.splitext(name)[0]

    # since we had to resize/pad, segmentation is not reliable anymore
    # discard the segmentation info
    if key in iris_map.keys():
        iris_map.pop(key)

    #     # draw
    #     print("New Coordinates (y, x, r):", iy, ix, ir)
    #     cv2.circle(newimg, (ix, iy), 2, (255, 255, 255))
    #     cv2.circle(newimg, (ix, iy), ir, (255, 255, 255))

    # # save a copy for posterior analysis
    # cv2.imwrite(
    #     '/afs/crc.nd.edu/user/a/akuehlka/tmp/printouts/{}'.format(os.path.basename(name)), newimg)

    return newimg, iris_data, iris_map


def read_single_image(fname, channel=1, dt=np.float32):

    imgs = []

    # for i, fname in enumerate(fnames):
    try:
        if 'gif' in str(os.path.splitext(fname)[1]).lower():
            if channel == 1:
                pil_img = Image.open(fname).convert("L")
            else:
                pil_img = Image.open(fname).convert("RGB")
            img = np.array(pil_img.getdata(), dtype=dt).reshape(pil_img.size[1], pil_img.size[0], channel)[:, :, ::-1]
        else:
            if channel == 1:
                img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
                img = img[:, :, np.newaxis]
            else:
                img = cv2.imread(fname, cv2.IMREAD_COLOR)
        img = np.array(img, dtype=dt)

        # imgs += [img]

    except Exception:
        raise (Exception, 'Can not read the image %s' % fname)

    # imgs = np.array(imgs, dtype=dt)
    imgs = np.array(img, dtype=dt)

    return imgs


def load_images_hdf5(fnames, channel=1, dsname='default', tmppath='/work/allansp/mcnns_tmp', segmentation_data=()):
    print('-- loading raw images ...')
    sys.stdout.flush()
    dt = np.float32
    imgsizes = []
    iris_data, iris_map = (np.array([]), np.array([]))
    if segmentation_data:
        iris_data = segmentation_data[0]
        iris_map = segmentation_data[1]

    # create a pytables file to contain all images
    # this way we'll be able to work with datasets that don't fit into the memory
    # based on https://kastnerkyle.github.io/posts/using-pytables-for-larger-than-ram-data-processing/
    hdf5_path = '{}/{}.hdf5'.format(tmppath, dsname)
    hdf5_lock = '{}/{}.lock'.format(tmppath, dsname)
    if not os.path.exists(hdf5_path):
        print("Cache not found, loading data...")
        os.system('echo "1" > ' + hdf5_lock)
        hdf5_file = tables.open_file(hdf5_path, mode='w')
        filters = tables.Filters(complevel=5, complib='blosc')
        npimg = read_single_image(fnames[0], dt=dt)
        data_storage = hdf5_file.create_earray(hdf5_file.root, 'imgs',
                                               tables.Atom.from_dtype(np.dtype(dt, npimg.shape)),
                                               shape=tuple([0] + list(npimg.shape)),
                                               # obj=npimg,
                                               filters=filters,
                                               expectedrows=len(fnames)
                                               )
        for i, fname in enumerate(fnames):
            # print('Reading ', fname, i)
            img = read_single_image(fname, channel=channel)
            imgsizes.append(img.shape)
            # print(fname, img.shape)
            if img.shape[0] != data_storage.shape[1] or img.shape[1] != data_storage.shape[2]:
                print("Resizing image", fname, "with shape", img.shape)
                # resize the image to fit the current size, and update segmentation info
                img, iris_data, iris_map = resizeProp(img, data_storage.shape[1:], fname,
                                                        segmentation_data=segmentation_data)
                # cv2.imwrite('imgs/img-%05d.png'%i, img[:, :, 0].astype(np.uint8))
            data_storage.append(img[np.newaxis, :, :, :])

        hdf5_file.close()
        os.system('rm ' + hdf5_lock)

    # check for lock file/wait
    while os.path.exists(hdf5_lock):
        print("Waiting for lock release...")
        time.sleep(5)
    data_storage = tables.open_file(hdf5_path, mode='r')

    return data_storage.root.imgs, iris_data, iris_map


def load_images(fnames, channel=1):
    print('-- loading raw images ...')
    sys.stdout.flush()
    dt = np.float32

    imgs = []

    for i, fname in enumerate(fnames):
        try:
            if 'gif' in str(os.path.splitext(fname)[1]).lower():
                if channel == 1:
                    pil_img = Image.open(fname).convert("L")
                else:
                    pil_img = Image.open(fname).convert("RGB")
                img = np.array(pil_img.getdata(), dtype=dt).reshape(pil_img.size[1], pil_img.size[0], channel)[:, :, ::-1]
            else:
                if channel == 1:
                    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
                    img = img[:, :, np.newaxis]
                else:
                    img = cv2.imread(fname, cv2.IMREAD_COLOR)
            img = np.array(img, dtype=dt)

            imgs += [img]

        except Exception:
            raise (Exception, 'Can not read the image %s' % fname)

    imgs = np.array(imgs, dtype=dt)

    return imgs


def __grouper(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    return it.zip_longest(fillvalue=fillvalue, *args)


def mosaic(n, imgs):
    """
    Make a grid from images.
    n    -- number of grid columns
    imgs -- images (must have same size and format)
    :param imgs:
    :param w:
    """
    imgs = iter(imgs)
    img0 = imgs.__next__()
    pad = np.zeros_like(img0)
    imgs = it.chain([img0], imgs)
    rows = __grouper(n, imgs, pad)
    return np.vstack(map(np.hstack, rows))


def read_csv_file(fname, sequenceid_col=1, delimiter=',', remove_header=True):

    csv_hash = {}
    csv_data = []

    with open(fname) as f:
        data = list(csv.reader(f, delimiter=delimiter))
        # -- removing header
        if remove_header:
            data = data[1:]
        for r_idx, row in enumerate(data):
            csv_data += [row]
            csv_hash[os.path.splitext(row[sequenceid_col])[0]] = r_idx
    csv_data = np.array(csv_data)

    return csv_data, csv_hash


def get_interesting_samples(ground_truth, scores, threshold, n_samples=1, label_neg=-1, label_pos=1):
    """
    Return the n most confusing positive and negative sample indexes. Positive samples have
    scores >= threshold and are labeled label_pos in ground_truth. Negative samples are labeled label_neg.
    @param ground_truth:
    @param scores:
    @param threshold:
    @param n_samples: number of samples to be saved
    @param label_neg:
    @param label_pos:
    """
    pos_hit = []
    neg_miss = []
    neg_hit = []
    pos_miss = []

    for idx, (gt, score) in enumerate(zip(ground_truth, scores)):
        if score >= threshold:
            if gt == label_pos:
                # -- positive hit
                pos_hit += [idx]
            else:
                # -- negative miss
                neg_miss += [idx]
        else:
            if gt == label_neg:
                # -- negative hit
                neg_hit += [idx]
            else:
                # -- positive miss
                pos_miss += [idx]

    # -- interesting samples
    scores_aux = np.empty(scores.shape, dtype=scores.dtype)

    scores_aux[:] = np.inf
    scores_aux[pos_hit] = scores[pos_hit]
    idx = min(n_samples, len(pos_hit))
    int_pos_hit = scores_aux.argsort()[:idx]

    scores_aux[:] = np.inf
    scores_aux[neg_miss] = scores[neg_miss]
    idx = min(n_samples, len(neg_miss))
    int_neg_miss = scores_aux.argsort()[:idx]

    scores_aux[:] = -np.inf
    scores_aux[neg_hit] = scores[neg_hit]
    idx = min(n_samples, len(neg_hit))
    if idx == 0:
        idx = -len(scores_aux)
    int_neg_hit = scores_aux.argsort()[-idx:]

    scores_aux[:] = -np.inf
    scores_aux[pos_miss] = scores[pos_miss]
    idx = min(n_samples, len(pos_miss))
    if idx == 0:
        idx = -len(scores_aux)
    int_pos_miss = scores_aux.argsort()[-idx:]

    r_dict = {'true_positive': int_pos_hit,
              'false_negative': int_neg_miss,
              'true_negative': int_neg_hit,
              'false_positive': int_pos_miss,
              }

    return r_dict


def create_mosaic(all_data, resize=False, max_axis=64, n_col=50, quality=50, output_fname='mosaic.jpg'):
    """ Create a mosaic.

    Args:
        all_data:
        resize:
        max_axis:
        n_col:
        quality:
        output_fname:

    Returns:

    """
    print('-- creating mosaic ...')
    sys.stdout.flush()

    alldata = []
    for idx in range(len(all_data)):
        img = all_data[idx]
        img = np.squeeze(img)
        if resize:
            ratio = max_axis / np.max(img.shape)
            new_shape = (int(img.shape[1] * ratio), int(img.shape[0] * ratio))
            alldata += [cv2.resize(img, new_shape)]
        else:
            alldata += [img]
    mosaic_img = mosaic(n_col, alldata)
    print('-- saving mosaic', output_fname)
    sys.stdout.flush()

    cv2.imwrite(output_fname, mosaic_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


def retrieve_fnames_by_type(input_path, file_type):

    dir_names = []
    for root, subFolders, files in os.walk(input_path):
        for f in files:
            if file_type in [os.path.splitext(f)[1], ".*"]:
                dir_names += [root]
                break

    dir_names = sorted(dir_names)

    fnames = []
    for dir_name in dir_names:
        dir_fnames = sorted(glob(os.path.join(input_path, dir_name, '*' + file_type)))
        fnames += dir_fnames

    return fnames


def classification_results_summary(report):
    """ This method is responsible for printing a summary of the classification results.

    Args:
        report (dict): A dictionary containing the measures (e.g., Acc, APCER)  computed for each test set.

    """

    print('-- Classification Results', flush=True)

    headers = ['Testing set', 'Threshold (value)', 'AUC', 'ACC', 'Balanced ACC', 'BPCER (FAR)', 'APCER (FRR)', 'HTER']
    header_line = "| {:<20s} | {:<20s} | {:<12s} | {:<12s} | {:<12s} | {:<12s} | {:<12s} | {:<8s} |\n".format(*headers)
    sep_line = '-' * (len(header_line) - 1) + '\n'

    final_report = sep_line
    final_report += header_line

    for k1 in sorted(report):
        final_report += sep_line
        for k2 in sorted(report[k1]):
            values = [k1,
                      "{} ({:.2f})".format(k2, report[k1][k2]['threshold']),
                      report[k1][k2]['auc'],
                      report[k1][k2]['acc'],
                      report[k1][k2]['bacc'],
                      report[k1][k2]['bpcer'],
                      report[k1][k2]['apcer'],
                      report[k1][k2]['hter'],
                      ]

            line = "| {:<20s} | {:<20s} | {:<12.4f} | {:<12.4f} | {:<12.4f} | {:<12.4f} | {:<12.4f} | {:<8.4f} |\n".format(*values)
            final_report += line

    final_report += sep_line

    print(final_report, flush=True)


def crop_img(img, cx, cy, max_axis, padding=0):
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

def get_iris_region(fnames, iris_location, max_axis):
    """

    Args:
        fnames (numpy.ndarray):

    Returns:

    """

    origin_imgs = load_images(fnames)
    iris_location_list, iris_location_hash = read_csv_file(iris_location, sequenceid_col=3)

    imgs = []
    for i, fname in enumerate(fnames):

        key = os.path.splitext(os.path.basename(fname))[0]

        try:
            cx, cy = iris_location_list[iris_location_hash[key]][-3:-1]
            cx = int(float(cx))
            cy = int(float(cy))
        except IndexError:
            print('Warning: Iris location not found. Cropping in the center of the image', fname)
            cy, cx = origin_imgs[i].shape[0]//2, origin_imgs[i].shape[1]//2

        img = crop_img(origin_imgs[i], cx, cy, max_axis)
        imgs += [img]

    imgs = np.array(imgs, dtype=np.uint8)

    return imgs

def get_imgs(fnames, iris_location, operation, max_axis):

    if 'crop' in operation:
        imgs = get_iris_region(fnames, iris_location, max_axis)
    else:
        imgs = load_images(fnames)

    return imgs


def convert_numpy_dict_items_to_list(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = convert_numpy_dict_items_to_list(v)
        else:
            new_dict = {}
            for k, v in d.items():
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                new_dict[k] = v
            return new_dict


def unique_everseen(iterable, key=None):
    # -- list unique elements, preserving order. Remember all elements ever seen
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in it.filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element
