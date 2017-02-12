# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
import cv2
import matplotlib.image as mpimg
import numpy as np

from lesson_functions import bin_spatial, color_hist, get_hog_features

color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 16  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off


def convert_color_space(img):
    if color_space != 'RGB':
        if color_space == 'HSV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    return img


def compute_hog_features(img):
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(img.shape[2]):
            hog_features.append(
                get_hog_features(img[:, :, channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
            )
        return np.array(hog_features)
    else:
        hog_features = get_hog_features(
            img[:, :, hog_channel], orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False
        )
    return hog_features


def extract_features(imgs):
    features = []
    for file in imgs:
        img = mpimg.imread(file)
        img = convert_color_space(img)
        img_features = single_img_features(img)
        features.append(img_features)

    return features


def search_windows(img, windows, clf, scaler, hog_features=None):
    on_windows = []
    for window in windows:
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

        hog_offset = (window[0][0] // pix_per_cell, window[0][1] // pix_per_cell)
        hog_width = (test_img.shape[1] // pix_per_cell) - 1

        features = single_img_features(
            test_img, hog_features=hog_features[:, hog_offset[1]:hog_offset[1] + hog_width, hog_offset[0]:hog_offset[0] + hog_width, :, :, :].ravel()
        )

        test_features = scaler.transform(features.reshape(1, -1))
        if clf.predict(test_features) == 1:
            on_windows.append(window)

    return on_windows


def single_img_features(img, hog_features=None):
    feature_image = np.copy(img)
    img_features = []

    if spatial_feat:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        img_features.append(spatial_features)

    if hist_feat:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        img_features.append(hist_features)

    if hog_features is not None:
        img_features.append(hog_features)

    return np.concatenate(img_features)
