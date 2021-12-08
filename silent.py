import numpy as np
import sys

import src.filters
import src.noises
import imageio
import cv2
import time
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import mean_squared_error


def rgb2gray(rgb):  # pass image from RGB to gray levels
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def filter_menu():
    print(" Choose up to three filter options (e.g. 2 1)")
    print("(1) Median filter")
    print("(2) Cut filter")
    print("(3) Low pass filter")
    print("(4) Band stop filter")

    options = [int(opt) for opt in input().split()]

    return options


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print('usage: python silent.py <INPUT IMAGE>')
        exit(0)

    img_orig = cv2.imread(sys.argv[1])
    img_true = cv2.imread(sys.argv[2])

    # Convert image to gray levels
    img_gray = rgb2gray(img_orig)
    img_true = cv2.cvtColor(img_true, cv2.COLOR_RGB2GRAY)

    print('Image dimensions:')
    print(img_gray.shape)

    # Choose and apply filter functions
    filter_options = filter_menu()

    filters = {
        1: src.filters.median,
        2: src.filters.cut,
        3: src.filters.low_pass,
        4: src.filters.bandstop,
    }

    # Apply chosen filters
    img_gray_filtered = np.copy(img_gray)
    img_true = cv2.resize(img_true, (500, 500), interpolation=cv2.INTER_LINEAR)

    filter_name = ['Median', 'Cut', 'Low pass', 'Bandstop']

    for filter_opt in filter_options:
        start = time.time()

        img_gray_filtered = filters[filter_opt](img_gray_filtered)
        cv2.imwrite('output_' + filter_name[filter_opt - 1] + '.png', img_gray_filtered)
        img_gray_filtered = cv2.resize(img_gray_filtered, (500, 500), interpolation=cv2.INTER_LINEAR)

        end = time.time()

        img_gray_filtered = img_gray_filtered.astype('uint8')
        img_true = img_true.astype('uint8')

        PSNR = peak_signal_noise_ratio(img_true, img_gray_filtered)
        SSIM = structural_similarity(img_true, img_gray_filtered)
        MSE = mean_squared_error(img_true, img_gray_filtered)

        print('============================== {} filter =============================='.format(
            filter_name[filter_opt - 1]))
        print('elapsed time: {}s'.format(end - start))
        print('PSNR: {}, SSIM: {}, MSE: {}'.format(PSNR, SSIM, MSE))
        print('=======================================================================')
        print('')

