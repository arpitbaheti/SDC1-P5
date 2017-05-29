import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2, sys
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
import pickle
from sklearn.externals import joblib

import fnmatch
import os

DEBUG = 0
IMAGE_PROCESS = 0

### Setting parameters gave best result to me.
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400, 656] # Min and max in y to search in slide_window()
overlap = 0.5
scale = 1.5

SVC_CLASSIFIER = 'LinearSVC_classifier.pkl'
X_SCALER_PKL = 'X_SCALER_svc.pkl'
SCALED_X_PKL = 'SCALED_X_svc.pkl'
Y_PKL = 'Y_svc.pkl'
VIDEO_FILE = 'project_video.mp4'

# make a list of images to read in
image_fnames = []
def walk_dirs(dirname):
    for root, dirnames, filenames in os.walk(dirname):
        for filename in fnmatch.filter(filenames, '*.png'):
            image_fnames.append(os.path.join(root, filename))
    return image_fnames

def get_image_names():
    global image_fnames
    image_fnames = walk_dirs('./data')

# Divide up into cars and notcars
cars = []
notcars = []
get_image_names()

for image_fname in image_fnames:
    if 'non-vehicles' in image_fname:
        notcars.append(image_fname)
    else:
        cars.append(image_fname)

if DEBUG == 1:
    car_ind = np.random.randint(0, len(cars))
    notcar_ind  = np.random.randint(0, len(notcars))

    car_image = mpimg.imread(cars[car_ind])
    notcar_image = mpimg.imread(notcars[notcar_ind])

    car_features, car_hog_image = single_img_features(car_image, color_space=color_space, spatial_size=spatial_size,
                            hist_bins=hist_bins, orient=orient,
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                            spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat, vis = True)

    notcar_features, notcar_hog_image =  single_img_features(notcar_image, color_space=color_space, spatial_size=spatial_size,
                            hist_bins=hist_bins, orient=orient,
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                            spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat, vis = True)

    images = [car_image, car_hog_image, notcar_image, notcar_hog_image]
    titles = ['car image', 'car hog image', 'notcar image', 'notcar hog image']
    fig = plt.figure(figsize=(12,3))
    visualize(fig, 1, 4, images, titles)


def train():
    t=time.time()
    if not os.path.exists(X_SCALER_PKL):
        car_features = extract_features(cars, color_space= color_space, spatial_size=spatial_size,
                                        hist_bins=hist_bins, orient=orient,
                                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                                        hist_feat=hist_feat, hog_feat=hog_feat)

        notcar_features = extract_features(notcars, color_space= color_space, spatial_size=spatial_size,
                                        hist_bins=hist_bins, orient=orient,
                                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                                        hist_feat=hist_feat, hog_feat=hog_feat)

        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    else:
        scaled_X, X_scaler, y = load_extracted_features()

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using spatial binning of:',spatial_size,
        'and', hist_bins,'histogram bins')
    print('Feature vector length:', len(X_train[0]))
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to extract feature')

    if not os.path.exists(SVC_CLASSIFIER):
        # Use a linear SVC
        svc = LinearSVC()
        # Check the training time for the SVC
        t=time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        joblib.dump(svc, SVC_CLASSIFIER)
        joblib.dump(X_scaler, X_SCALER_PKL)
        joblib.dump(scaled_X, SCALED_X_PKL)
        joblib.dump(y, Y_PKL)
    else:
        svc = laod_pretrained_model()

    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

    return svc, X_scaler


def laod_pretrained_model():
    print ("Restoring saved model  " + SVC_CLASSIFIER);
    svc = joblib.load(SVC_CLASSIFIER)
    return svc

def load_extracted_features():
    print ("Restoring feature and label " + X_SCALER_PKL + ' ' + SCALED_X_PKL + " " +Y_PKL);
    scaled_X = joblib.load(SCALED_X_PKL)
    X_scaler = joblib.load(X_SCALER_PKL)
    y = joblib.load(Y_PKL)
    return scaled_X, X_scaler, y


def detect_car_in_image(image, svc, X_scaler):
    t =time.time()
    draw_image = np.copy(image)
    image = image.astype(np.float32)/255

    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                    xy_window=(96, 96), xy_overlap=(overlap, overlap))


    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)

    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    t2 = time.time()

    print(round(t2-t, 5), 'Seconds to detect ', len(windows), 'windows')

    return window_img

def detect_car_in_scaled_image(image, svc, X_scaler):
    t =time.time()
    count = 0
    draw_image = np.copy(image)
    image = image.astype(np.float32)/255

    # make a heat map of zeros
    heatmap = np.zeros_like(image[:,:,0]).astype(np.float)

    img_tosearch = image[y_start_stop[0]: y_start_stop[1],:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            count += 1
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()

            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_image,(xbox_left, ytop_draw + y_start_stop[0]),(xbox_left + win_draw, ytop_draw + win_draw + y_start_stop[0]),(0,0,255),6)
                heatmap[ytop_draw + y_start_stop[0] : ytop_draw + win_draw + y_start_stop[0], xbox_left: xbox_left + win_draw] += 1

    t2 = time.time()

    return draw_image, heatmap


def process_image(image):
    draw_image, heatmap = detect_car_in_scaled_image(image, svc, X_scaler)

    #heatmap = apply_threshold(heatmap, 2)

    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    return draw_img

def main():
    global svc, X_scaler
    svc, X_scaler = train()

    if IMAGE_PROCESS == 1:

        images = glob.glob('test_images/*')
        draw_images = []
        titles = []
        for i , image_path in enumerate(images):
            image = mpimg.imread(image_path)
            #draw_image = detect_car_in_image(image, svc, X_scaler)
            draw_image, heatmap = detect_car_in_scaled_image(image, svc, X_scaler)
            heatmap = apply_threshold(heatmap, 2)
            labels = label(heatmap)
            draw_img = draw_labeled_bboxes(np.copy(image), labels)
            draw_images.append(draw_img)
            draw_images.append(heatmap)
            titles.append(' ')
            titles.append(' ')

        fig = plt.figure(figsize=(12,24))
        visualize(fig, 8, 2, draw_images, titles)
        plt.show()

    else:

        test_output = 'test.mp4'

        clip = VideoFileClip(VIDEO_FILE)
        test_clip = clip.fl_image(process_image)

        test_clip.write_videofile(test_output, audio=False)

if __name__ == '__main__':
    main()
