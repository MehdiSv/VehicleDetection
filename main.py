import glob
import os
import time

import matplotlib.pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip
from scipy.ndimage.measurements import label
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from heatmapper import Heatmapper
from lesson_functions import *
from solution import search_windows, extract_features, compute_hog_features, convert_color_space

SCALER_PICKLE_FILENAME = 'scaler.joblib.pkl'
SCALED_X_PICKLE_FILENAME = 'scaled_x.joblib.pkl'
SVC_PICKLE_FILENAME = 'svc.joblib.pkl'

cars = []
notcars = []

car_features = None
notcar_features = None
X_scaler = None
scaled_X = None
svc = None


def compute_features():
    global car_features, notcar_features
    car_features = extract_features(cars)
    notcar_features = extract_features(notcars)


def learn_scale():
    global car_features, notcar_features, X_scaler, scaled_X
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    joblib.dump(X_scaler, SCALER_PICKLE_FILENAME, compress=9)
    scaled_X = X_scaler.transform(X)
    joblib.dump(scaled_X, SCALED_X_PICKLE_FILENAME, compress=9)


def learn():
    global car_features, notcar_features, scaled_X, svc
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    rand_state = np.random.randint(0, 100)
    x_train, x_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
    svc = LinearSVC()
    t = time.time()
    svc.fit(x_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    print('Test Accuracy of SVC = ', round(svc.score(x_test, y_test), 4))
    joblib.dump(svc, SVC_PICKLE_FILENAME, compress=9)
    return svc


def process_image(img):
    global windows
    draw_image = np.copy(img)
    img = img.astype(np.float32) / 255
    img = convert_color_space(img)

    hog_features = compute_hog_features(img)

    hot_windows = search_windows(img, windows, svc, X_scaler, hog_features=hog_features)

    heatmapper.add_frame_boxes(hot_windows)
    heatmap = heatmapper.compute_heatmap(img)

    labels = label(heatmap)

    return draw_labeled_bboxes(draw_image, labels)


def draw_labeled_bboxes(img, labels):
    for car_number in range(1, labels[1] + 1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    return img


images = glob.glob('data/**/*.png', recursive=True)
for image in images:
    if 'non-vehicles' in image:
        notcars.append(image)
    else:
        cars.append(image)

if os.path.isfile(SCALER_PICKLE_FILENAME):
    print('Loader scaler from pickle...')
    X_scaler = joblib.load(SCALER_PICKLE_FILENAME)
    print('Done')
else:
    print('Learning scale from scratch...')
    if car_features is None:
        compute_features()

    learn_scale()
    print('Done')

if os.path.isfile(SVC_PICKLE_FILENAME):
    print('Loading SVC from pickle...')
    svc = joblib.load(SVC_PICKLE_FILENAME)
    print('Done')
else:
    print('Learning SVC from scratch...')
    if car_features is None:
        compute_features()
    if scaled_X is None:
        scaled_X = joblib.load(SCALED_X_PICKLE_FILENAME)

    learn()
    print('Done')

window_sizes = [
    64,
    96,
    128,
]

y_start_stop = [
    [50, 70],
    [50, 80],
    [50, 90],
]

windows = []
for i in range(len(window_sizes)):
    windows += slide_window(
        [1280, 720], x_start_stop=[None, None], y_start_stop_percent=y_start_stop[i],
        xy_window=(window_sizes[i], window_sizes[i]), xy_overlap=(0.8, 0.8)
    )

heatmapper = Heatmapper()
video = 'project_video'
white_output = '{}_test.mp4'.format(video)
clip = VideoFileClip('../CarND-Advanced-Lane-Lines/project_video.mp4'.format(video)).subclip(39, 40)
white_clip = clip.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)
