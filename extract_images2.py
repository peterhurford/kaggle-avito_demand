# HT: https://www.kaggle.com/c/avito-demand-prediction/discussion/59414#347781
# HT: https://www.kaggle.com/khahuras/advanced-image-features-extraction/notebook

import os
import operator

from ast import literal_eval

import pathos.multiprocessing as mp

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import cv2
import imutils
from PIL import Image, ImageStat
from skimage import feature, measure
from imutils import contours

from scipy.stats import kurtosis, skew, itemfreq
from scipy.ndimage import laplace, sobel

from utils import print_step
from cache import get_data, is_in_cache, load_cache, save_in_cache


def get_image(image_path):
    try:
        cv_img = cv2.imread(image_path)
        cv_bw_img = cv2.imread(image_path, 0)
        cv_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        return (None, cv_img, image_path, cv_bw_img, cv_gray)
    except (OSError, cv2.error):
        return False


def get_keyp(img):
    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(img, None)
    num_kp = len(kp)
    kp_x_mean = np.array([k.pt[0] for k in kp]).mean()
    kp_x_std = np.array([k.pt[0] for k in kp]).std()
    kp_y_mean = np.array([k.pt[1] for k in kp]).mean()
    kp_y_std = np.array([k.pt[1] for k in kp]).std()
    return [num_kp, kp_x_mean, kp_x_std, kp_y_mean, kp_y_std]


def get_edges(gray):
    edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
    edges2 = np.array([(edge > 0) for edge in edges]).astype('int')
    edges_mean = edges2.mean()
    edges_sum_mean = edges2.sum(axis=1).mean()
    edges_std = edges2.std()
    edges_sum_std = edges2.sum(axis=1).std()
    edges_skew = skew(edges2, axis=1).mean()
    edges_kurtosis = kurtosis(edges2, axis=1).mean()
    return edges, edges_mean, edges_sum_mean, edges_std, edges_sum_std, edges_skew, edges_kurtosis


def get_lines(edges):
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    if lines is not None:
        lines_num = len(lines)
        lines_theta_mean = np.array([l[0][0] for l in lines]).mean()
        lines_theta_std = np.array([l[0][0] for l in lines]).std()
        lines_rho_mean = np.array([l[0][1] for l in lines]).mean()
        lines_rho_std = np.array([l[0][1] for l in lines]).std()
        return lines_num, lines_theta_mean, lines_theta_std, lines_rho_mean, lines_rho_std
    else:
        return 0, 0, 0, 0, 0


def get_bright_spots(gray):
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    labels = measure.label(thresh, neighbors=8, background=0)
    mask = np.zeros(thresh.shape, dtype='uint8')
    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(thresh.shape, dtype='uint8')
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        if numPixels > 300:
            mask = cv2.add(mask, labelMask)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    if len(cnts):
        cnts = contours.sort_contours(cnts)[0]
        num_bright_spots = cnts[0].shape[0]
        return num_bright_spots
    else:
        return 0


def get_density(img, gray):
    return cv2.countNonZero(gray) / img.size


def get_contours(gray):
    ret, thresh = cv2.threshold(gray, 127, 255,0)
    cnts = cv2.findContours(thresh, 2, 1)
    return cnts[0]


def get_moments(cnts):
    return cv2.moments(cnts[0])


def get_aspect_ratio(cnts):
    x, y, w, h = cv2.boundingRect(cnts)
    aspect_ratio = float(w) / h


def get_median_noise(img):
    median = cv2.medianBlur(img, 5)
    diff = img - median
    return [diff.sum(), diff.mean(), diff.std(), skew(diff, axis=1).mean(), kurtosis(diff, axis=1).mean()]

def get_gaussian_noise(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    diff = img - blur
    return [diff.sum(), diff.mean(), diff.std(), skew(diff, axis=1).mean(), kurtosis(diff, axis=1).mean()]

def get_clahe_noise(img):
    clahe = cv2.createCLAHE()
    clahe = clahe.apply(img)
    diff = img - clahe
    return [diff.sum(), diff.mean(), diff.std(), skew(diff, axis=1).mean(), kurtosis(diff, axis=1).mean()]


def get_data_from_image(dat, core=0, i=0):
    img_path = dat[2].replace('train_jpg/', '').replace('test_jpg/', '')
    num_kp, kp_x_mean, kp_x_std, kp_y_mean, kp_y_std = get_keyp(dat[3])
    edges, edges_mean, edges_sum_mean, edges_std, edges_sum_std, edges_skew, edges_kurtosis = get_edges(dat[4])
    lines_num, lines_theta_mean, lines_theta_std, lines_rho_mean, lines_rho_std = get_lines(edges)
    num_bright_spots = get_bright_spots(dat[4])
    img_density = get_density(dat[1], dat[4])
    contours = get_contours(dat[4])
    aspect_ratio = get_aspect_ratio(contours)
    moments = list(get_moments(contours).values())
    median_noise_sum, median_noise_mean, median_noise_std, median_noise_skew, median_noise_kurtosis = get_median_noise(dat[1])
    gaussian_noise_sum, gaussian_noise_mean, gaussian_noise_std, gaussian_noise_skew, gaussian_noise_kurtosis = get_gaussian_noise(dat[1])
    clahe_noise_sum, clahe_noise_mean, clahe_noise_std, clahe_noise_skew, clahe_noise_kurtosis = get_clahe_noise(dat[3])
    return [img_path, num_kp, kp_x_mean, kp_x_std, kp_y_mean, kp_y_std, edges_mean, edges_sum_mean, edges_std, edges_sum_std, edges_skew, edges_kurtosis, lines_num, lines_theta_mean, lines_theta_std, lines_rho_mean, lines_rho_std, num_bright_spots, img_density, aspect_ratio] + moments + [median_noise_sum, median_noise_mean, median_noise_std, median_noise_skew, median_noise_kurtosis, gaussian_noise_sum, gaussian_noise_mean, gaussian_noise_std, gaussian_noise_skew, gaussian_noise_kurtosis, clahe_noise_sum, clahe_noise_mean, clahe_noise_std, clahe_noise_skew, clahe_noise_kurtosis]


def data_to_df(data):
    columns = ['img_path', 'num_kp', 'kp_x_mean', 'kp_x_std', 'kp_y_mean', 'kp_y_std', 'edges_mean', 'edges_sum_mean', 'edges_std', 'edges_sum_std', 'edges_skew', 'edges_kurtosis', 'lines_num', 'lines_theta_mean', 'lines_theta_std', 'lines_rho_mean', 'lines_rho_std', 'num_bright_spots', 'img_density', 'img_aspect_ratio', 'img_moment_m11', 'img_moment_nu21', 'img_moment_mu02', 'img_moment_mu21', 'img_moment_mu30', 'img_moment_nu20', 'img_moment_nu02', 'img_moment_nu12', 'img_moment_m00', 'img_moment_m10', 'img_moment_mu20', 'img_moment_m30', 'img_moment_nu03', 'img_moment_mu03', 'img_moment_m20', 'img_moment_m02', 'img_moment_mu12', 'img_moment_m03', 'img_moment_m12', 'img_moment_m01', 'img_moment_mu11', 'img_moment_m21', 'img_moment_nu11', 'img_moment_nu30', 'median_noise_sum', 'median_noise_mean', 'median_noise_std', 'median_noise_skew', 'median_noise_kurtosis', 'gaussian_noise_sum', 'gaussian_noise_mean', 'gaussian_noise_std', 'gaussian_noise_skew', 'gaussian_noise_kurtosis', 'clahe_noise_sum', 'clahe_noise_mean', 'clahe_noise_std', 'clahe_noise_skew', 'clahe_noise_kurtosis']
    return pd.DataFrame(data, columns = columns)

print_step('Loading images')
train_images_dir = os.path.expanduser('train_jpg')
train_image_files = [x.path for x in os.scandir(train_images_dir)]
test_images_dir = os.path.expanduser('test_jpg')
test_image_files = [x.path for x in os.scandir(test_images_dir)]
image_files = train_image_files + test_image_files
print_step('Loaded %d images' % len(image_files))


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def get_img_data(index, image_files):
    print_step('[Core %d] Start' % index)
    if not is_in_cache('img_data_' + str(index)):
        data = []
        i = 0
        for image_file in image_files:
            dat = get_image(image_file)
            if dat:
                data += [get_data_from_image(dat, core=index, i=i)]
            i += 1
            if i % 50 == 0:
                print_step('[Core %d] Completed %d / %d...' % (index, i, len(image_files)))
        print_step('[Core %d] Done. Saving...' % index)
        save_in_cache('img_data_' + str(index), data_to_df(data), None)
    else:
        print(str(index) + ' already in cache! Skipping...')
    return True

n_cpu = mp.cpu_count()
n_nodes = 70

print_step('Chunking images')
image_chunks = list(chunks(image_files, 1000))
print_step('Chunked images into %d groups with %d images per group' % (len(image_chunks), len(image_chunks[0])))

pool = mp.ProcessingPool(n_nodes, maxtasksperchild=500)
print_step('Starting a jobs server with %d nodes' % n_nodes)
res = pool.map(lambda dat: get_img_data(dat[0], dat[1]), enumerate(image_chunks))
pool.close()
pool.join()
pool.terminate()
pool.restart()

print('~~~~~~~~~~~~~~~~')
print_step('Merging 1/9')
dfs = pool.map(lambda c: load_cache('img_data_' + str(c)), range(len(image_chunks)))
print_step('Merging 2/9')
dfs = map(lambda x: x[0], dfs)
print_step('Merging 3/9')
merge = pd.concat(dfs)
print_step('Merging 4/9')
train, test = get_data()
print_step('Merging 5/9')
merge['img_path'] = merge['img_path'].apply(lambda x: x.replace('test_jpg/', ''))
print_step('Merging 6/9')
merge['img_path'] = merge['img_path'].apply(lambda x: x.replace('train_jpg/', ''))
print_step('Merging 7/9')
merge['img_path'] = merge['img_path'].apply(lambda x: x.replace('.jpg', ''))
print_step('Merging 8/9')
train2 = train.merge(merge, left_on='image', right_on='img_path', how='left')
print_step('Merging 9/9')
test2 = test.merge(merge, left_on='image', right_on='img_path', how='left')

print_step('Dropping 1/2')
drops = list(set(train2.columns.values) - set(merge.columns.values) - {'deal_probability', 'item_id'})
drops += ['img_aspect_ratio', 'img_moment_m11', 'img_moment_mu21', 'img_moment_mu30', 'img_moment_nu02', 'img_moment_nu12', 'img_moment_m00', 'img_moment_m10', 'img_moment_mu03', 'img_moment_mu12', 'img_moment_m03', 'img_moment_m12', 'img_moment_mu11', 'img_moment_nu11', 'img_moment_mu20', 'img_moment_m20', 'img_moment_nu30', 'img_moment_m01', 'img_moment_m30', 'img_moment_nu03', 'img_moment_mu02', 'img_moment_m21', 'img_moment_nu21', 'img_moment_m02', 'img_moment_nu20']
train2.drop(drops, axis=1, inplace=True)
print_step('Dropping 2/2')
test2.drop(drops, axis=1, inplace=True)

print_step('Saving...')
save_in_cache('img_data2', train2, test2)
