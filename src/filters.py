import numpy as np
from scipy.fftpack import fftn, ifftn, fftshift
from scipy.spatial.distance import euclidean
import imageio
from matplotlib.colors import LogNorm


def _low_pass_filter(radius, final_shape):
    filt = np.zeros((radius, radius))

    # Create circle mask (pad later)
    for x in range(radius):
        for y in range(radius):
            if (radius // 2 - x) ** 2 + (radius // 2 - y) ** 2 < (radius // 2) ** 2:
                filt[x][y] = 1

    #print(filt.shape)

    # Calculate padding shape (rows)
    aux1 = final_shape[0] - filt.shape[0]
    if (aux1 % 2 == 0):  # even vs odd problem
        pad_rows = ((final_shape[0] - filt.shape[0]) // 2, \
                    (final_shape[0] - filt.shape[0]) // 2)
    else:
        pad_rows = ((final_shape[0] - filt.shape[0]) // 2 + 1, \
                    (final_shape[0] - filt.shape[0]) // 2 + 1)

    # Calculate padding shape (cols)
    aux2 = final_shape[1] - filt.shape[1]
    if (aux2 % 2 == 0):
        pad_cols = ((final_shape[1] - filt.shape[1]) // 2, \
                    (final_shape[1] - filt.shape[1]) // 2)
    else:
        pad_cols = ((final_shape[1] - filt.shape[1]) // 2 + 1, \
                    (final_shape[1] - filt.shape[1]) // 2 + 1)

    pad_shape = (pad_rows, pad_cols)

    # Apply padding
    filt = np.pad(filt, pad_shape, 'constant', constant_values=0)
    if (aux1 % 2 != 0 and aux2 % 2 != 0):
        filt = filt[0:filt.shape[0] - 1, 0:filt.shape[1] - 1]
    elif (aux1 % 2 != 0 and aux2 % 2 == 0):
        filt = filt[0:filt.shape[0] - 1, 0:filt.shape[1]]
    elif (aux1 % 2 == 0 and aux2 % 2 != 0):
        filt = filt[0:filt.shape[0], 0:filt.shape[1] - 1]

    #print(filt.shape)

    return filt


def low_pass(img, radius=501, debug=False):
    """
    Filters image using low pass filter in Fourier domain
    Returns filtered image
    """

    img_fft = fftn(img)
    img_fft_shift = fftshift(img_fft)

    filter_mask = _low_pass_filter(radius, img.shape)

    img_fft_shift_filtered = img_fft_shift * filter_mask

    # Generate result image
    return np.abs(ifftn(fftshift(img_fft_shift_filtered)))


def _local_median(img, x, y, k):
    """
    Computes median for k-neighborhood of img[x,y]
    """
    flat = img[x - k: x + k + 1, y - k: y + k + 1].flatten()
    flat.sort()
    return flat[len(flat) // 2]


def median(img, k=3):
    """
    Changes every pixel to the median of its neighboors
    """
    res = np.copy(img)

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if (x - k >= 0 and x + k < img.shape[0]) and \
                    (y - k >= 0 and y + k < img.shape[1]):
                res[x, y] = _local_median(img, x, y, k)

    return res


def cut(img):
    """
    Applies central horizontal threshold in Fourier spectrum
    """

    # Apply fourier transform and shift
    img_fft = fftn(img)
    img_fft_shift = fftshift(img_fft)

    # Filter image: remove upper and lower horizontal thirds (1/3)
    img_fft_shift_filtered = np.copy(img_fft_shift)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if ((x < img.shape[0] // 2 - img.shape[0] // 30 or \
                 x > img.shape[0] // 2 + img.shape[0] // 30) and \
                    (y < img.shape[1] // 2 - img.shape[1] // 30 or \
                     y > img.shape[1] // 2 + img.shape[1] // 30)):
                img_fft_shift_filtered[x, y] = 0
            if ((x < img.shape[0] // 3 or \
                 x > img.shape[0] * 2 // 3) or \
                    (y < img.shape[1] // 3 or \
                     y > img.shape[1] * 2 // 3)):
                img_fft_shift_filtered[x, y] = 0

    # Return to space domain result image using inverse
    return np.abs(ifftn(fftshift(img_fft_shift_filtered)))


def bandstop(img, threshold=0.0001):
    """
    Apply bandstop filter on Fourier spectrum
    """

    img_fft = fftn(img)

    # chose ther borders values
    borders = [
        img_fft[0, 0],
        img_fft[0, img.shape[1] - 1],
        img_fft[img.shape[0] - 1, 0],
        img_fft[img.shape[0] - 1, img.shape[1] - 1]
    ]

    # select the max value and create the treshold
    borders = np.array(borders)
    max_value = np.max(borders)
    T = threshold * np.abs(max_value)
    #print("Max: ", np.max(img_fft))
    #print("Min: ", np.min(img_fft))
    #print("Threshold: ", T)
    img_fft_shifted = fftshift(img_fft)

    center_x = img_fft_shifted.shape[0] // 2
    center_y = img_fft_shifted.shape[1] // 2
    radius = img_fft_shifted.shape[0] // 6

    # Apply the threshold out of the center of image
    for x in range(img_fft_shifted.shape[0]):
        for y in range(img_fft_shifted.shape[1]):
            if (center_x - x) ** 2 + (center_y - y) ** 2 > (radius) ** 2 and np.abs(img_fft_shifted[x, y]) > T:
                img_fft_shifted[x, y] = 0

    return np.abs(ifftn(fftshift(img_fft_shifted)))