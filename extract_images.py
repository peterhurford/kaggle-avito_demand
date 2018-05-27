# HT: https://www.kaggle.com/shivamb/ideas-for-image-features-and-image-quality/notebook
# HT: https://www.kaggle.com/classtag/extract-avito-image-features-via-keras-vgg16/notebook
# HT: https://www.kaggle.com/the1owl/natural-growth-patterns-fractals-of-nature
# HT: https://www.kaggle.com/cttsai/ensembling-gbms-lb-203

import os
import operator

from ast import literal_eval

import pathos.multiprocessing as mp

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

import cv2
from PIL import Image, ImageStat
from skimage import feature
from scipy.stats import kurtosis, skew
from scipy.ndimage import laplace, sobel

from scipy.stats import itemfreq

from utils import print_step
from cache import get_data, is_in_cache, load_cache, save_in_cache


def get_image(image_path):
    try:
        img = Image.open(image_path)
        cv_img = cv2.imread(image_path)
        cv_bw_img = cv2.imread(image_path, 0)
        return (img, cv_img, image_path, cv_bw_img)
    except OSError:
        return False


def get_dullness(img):
    # obtain the color palatte of the image 
    palatte = defaultdict(int)
    for pixel in img.getdata():
        palatte[pixel] += 1
    
    # sort the colors present in the image 
    sorted_x = sorted(palatte.items(), key=operator.itemgetter(1), reverse = True)
    light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25
    for i, x in enumerate(sorted_x[:pixel_limit]):
        if all(xx <= 20 for xx in x[0][:3]): ## dull : too much darkness 
            dark_shade += x[1]
        if all(xx >= 240 for xx in x[0][:3]): ## bright : too much whiteness 
            light_shade += x[1]
        shade_count += x[1]
        
    light_percent = round((float(light_shade)/shade_count) * 100, 2)
    dark_percent = round((float(dark_shade)/shade_count) * 100, 2)
    return light_percent, dark_percent

def dullness(im):
    # cut the images into two halves as complete average may give bias results
    halves = (im.size[0] / 2, im.size[1] / 2)
    im1 = im.crop((0, 0, im.size[0], halves[1]))
    im2 = im.crop((0, halves[1], im.size[0], im.size[1]))
    light_percent1, dark_percent1 = get_dullness(im1)
    light_percent2, dark_percent2 = get_dullness(im2)
    light_percent = (light_percent1 + light_percent2) / 2.0
    dark_percent = (dark_percent1 + dark_percent2) / 2.0 
    return light_percent, dark_percent


def average_pixel_width(im):
    im_array = np.asarray(im.convert(mode='L'))
    edges_sigma1 = feature.canny(im_array, sigma=3)
    apw = (float(np.sum(edges_sigma1)) / (im.size[0]*im.size[1]))
    return apw * 100


def color_array_to_label(x):
    return str(int(round(x[2] / 40) * 40)) + '_' + str(int(round(x[1] / 40) * 40)) + '_' + str(int(round(x[0] / 40) * 40))


def get_average_color(img):
    average_color = [img[:, :, i].mean() for i in range(img.shape[-1])]
    return average_color


def get_file_size(img_path):
    return os.stat(img_path).st_size


def get_blurrness_score(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(image, cv2.CV_64F).var()
    return fm


def get_data_from_image(dat, core=0, i=0):
    img_path = dat[2].replace('train_jpg/', '').replace('test_jpg/', '')
    img_size = [dat[0].size[0], dat[0].size[1]]
    (means, stds) = cv2.meanStdDev(dat[1])
    color_stats = np.concatenate([means, stds]).flatten()
    img_file_size = get_file_size(dat[2])
    im_stats_ = ImageStat.Stat(dat[0])
    sum_color = im_stats_.sum
    std_color = im_stats_.stddev
    rms_color = im_stats_.rms
    var_color = im_stats_.var
    mean_color = np.mean(dat[1].flatten())
    thing1 = (dat[3] < 10).sum()
    thing2 = (dat[3] > 245).sum()
    sobel00 = sobel(np.squeeze(dat[1][:, :, 0]), axis=0, mode='reflect', cval=0.0).ravel().var()
    sobel10 = sobel(np.squeeze(dat[1][:, :, 0]), axis=0, mode='reflect', cval=0.0).ravel().var()
    sobel20 = sobel(np.squeeze(dat[1][:, :, 0]), axis=0, mode='reflect', cval=0.0).ravel().var()
    sobel01 = sobel(np.squeeze(dat[1][:, :, 0]), axis=1, mode='reflect', cval=0.0).ravel().var()
    sobel11 = sobel(np.squeeze(dat[1][:, :, 0]), axis=1, mode='reflect', cval=0.0).ravel().var()
    sobel21 = sobel(np.squeeze(dat[1][:, :, 0]), axis=1, mode='reflect', cval=0.0).ravel().var()
    kurtosis_ = kurtosis(dat[1].ravel())
    skew_ = skew(dat[1].ravel())
    color_histogram = list(cv2.calcHist([dat[3]], [0], None, [256], [0, 256]).flatten())
    light_percent, dark_percent = dullness(dat[0])
    blur = cv2.Laplacian(dat[3], cv2.CV_64F).var()
    apw = average_pixel_width(dat[0])
    average_color = get_average_color(dat[1])
    average_red = average_color[0] / 255.0
    average_green = average_color[1] / 255.0
    average_blue = average_color[2] / 255.0
    average_color = color_array_to_label(average_color)
    return ([img_path] + img_size + [img_file_size, mean_color, std_color, sum_color, rms_color, var_color] + color_histogram + [thing1, thing2] +
            [sobel00, sobel10, sobel20, sobel01, sobel11, sobel21] + [kurtosis_, skew_] + [light_percent, dark_percent] + [blur] + [apw] + color_stats.tolist() +
            [average_red, average_green, average_blue, average_color])


def data_to_df(data):
    columns = (['img_path', 'img_size_x', 'img_size_y', 'img_file_size', 'img_mean_color', 'img_std_color', 'img_sum_color', 'img_rms_color', 'img_var_color'] +
               ['thing1', 'thing2'] + ['img_histogram_' + str(i) for i in range(256)] + ['img_sobel00', 'img_sobel10', 'img_sobel20', 'img_sobel01', 'img_sobel11', 'img_sobel21'] +
               ['img_kurtosis', 'img_skew', 'img_dullness_light_percent', 'img_dullness_dark_percent', 'img_blur', 'img_average_pixel_width', 'img_blue_mean', 'img_green_mean'] +
               ['img_red_mean', 'img_blue_std', 'img_green_std', 'img_red_std', 'img_average_red', 'img_average_green', 'img_average_blue', 'img_average_color'])
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
            if i % 100 == 0:
                print_step('[Core %d] Completed %d / %d...' % (index, i, len(image_files)))
        print_step('[Core %d] Done. Saving...' % index)
        save_in_cache('img_data_' + str(index), data_to_df(data), None)
    else:
        print(str(index) + ' already in cache! Skipping...')
    return True

n_cpu = mp.cpu_count()
n_nodes = 60

print_step('Chunking images')
image_chunks = list(chunks(image_files, 2000))
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
train2.drop(drops, axis=1, inplace=True)
print_step('Dropping 2/2')
test2.drop(drops, axis=1, inplace=True)

# Ooops... have to fix this in post.
print_step('Fixing 1/24')
train2['img_sum_color0'] = train2['img_sum_color'].apply(lambda x: literal_eval(x)[0] if str(x) != 'nan' else 0)
print_step('Fixing 2/24')
train2['img_sum_color1'] = train2['img_sum_color'].apply(lambda x: literal_eval(x)[1] if str(x) != 'nan' else 1)
print_step('Fixing 3/24')
train2['img_sum_color2'] = train2['img_sum_color'].apply(lambda x: literal_eval(x)[2] if str(x) != 'nan' else 2)
print_step('Fixing 4/24')
train2['img_var_color0'] = train2['img_var_color'].apply(lambda x: literal_eval(x)[0] if str(x) != 'nan' else 0)
print_step('Fixing 5/24')
train2['img_var_color1'] = train2['img_var_color'].apply(lambda x: literal_eval(x)[1] if str(x) != 'nan' else 1)
print_step('Fixing 6/24')
train2['img_var_color2'] = train2['img_var_color'].apply(lambda x: literal_eval(x)[2] if str(x) != 'nan' else 2)
print_step('Fixing 7/24')
train2['img_std_color0'] = train2['img_std_color'].apply(lambda x: literal_eval(x)[0] if str(x) != 'nan' else 0)
print_step('Fixing 8/24')
train2['img_std_color1'] = train2['img_std_color'].apply(lambda x: literal_eval(x)[1] if str(x) != 'nan' else 1)
print_step('Fixing 9/24')
train2['img_std_color2'] = train2['img_std_color'].apply(lambda x: literal_eval(x)[2] if str(x) != 'nan' else 2)
print_step('Fixing 10/24')
train2['img_rms_color0'] = train2['img_rms_color'].apply(lambda x: literal_eval(x)[0] if str(x) != 'nan' else 0)
print_step('Fixing 11/24')
train2['img_rms_color1'] = train2['img_rms_color'].apply(lambda x: literal_eval(x)[1] if str(x) != 'nan' else 1)
print_step('Fixing 12/24')
train2['img_rms_color2'] = train2['img_rms_color'].apply(lambda x: literal_eval(x)[2] if str(x) != 'nan' else 2)
print_step('Fixing 13/24')
test2['img_sum_color0'] = test2['img_sum_color'].apply(lambda x: literal_eval(x)[0] if str(x) != 'nan' else 0)
print_step('Fixing 14/24')
test2['img_sum_color1'] = test2['img_sum_color'].apply(lambda x: literal_eval(x)[1] if str(x) != 'nan' else 1)
print_step('Fixing 15/24')
test2['img_sum_color2'] = test2['img_sum_color'].apply(lambda x: literal_eval(x)[2] if str(x) != 'nan' else 2)
print_step('Fixing 16/24')
test2['img_var_color0'] = test2['img_var_color'].apply(lambda x: literal_eval(x)[0] if str(x) != 'nan' else 0)
print_step('Fixing 17/24')
test2['img_var_color1'] = test2['img_var_color'].apply(lambda x: literal_eval(x)[1] if str(x) != 'nan' else 1)
print_step('Fixing 18/24')
test2['img_var_color2'] = test2['img_var_color'].apply(lambda x: literal_eval(x)[2] if str(x) != 'nan' else 2)
print_step('Fixing 19/24')
test2['img_std_color0'] = test2['img_std_color'].apply(lambda x: literal_eval(x)[0] if str(x) != 'nan' else 0)
print_step('Fixing 20/24')
test2['img_std_color1'] = test2['img_std_color'].apply(lambda x: literal_eval(x)[1] if str(x) != 'nan' else 1)
print_step('Fixing 21/24')
test2['img_std_color2'] = test2['img_std_color'].apply(lambda x: literal_eval(x)[2] if str(x) != 'nan' else 2)
print_step('Fixing 22/24')
test2['img_rms_color0'] = test2['img_rms_color'].apply(lambda x: literal_eval(x)[0] if str(x) != 'nan' else 0)
print_step('Fixing 23/24')
test2['img_rms_color1'] = test2['img_rms_color'].apply(lambda x: literal_eval(x)[1] if str(x) != 'nan' else 1)
print_step('Fixing 24/24')
test2['img_rms_color2'] = test2['img_rms_color'].apply(lambda x: literal_eval(x)[2] if str(x) != 'nan' else 2)
print_step('Saving...')
save_in_cache('img_data', train2, test2)
