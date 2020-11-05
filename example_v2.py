import numpy as np
import cv2 as cv
import argparse
import math
import pickle
import matplotlib.pyplot as plt
import os
import sys

def resize_to_fit(frame, size):
    desired_width, desired_height = size
    actual_height, actual_width = tuple(frame.shape[:2])
    factor = min(desired_width / actual_width, desired_height / actual_height)
    return cv.resize(frame, (0,0), fx=factor, fy=factor)

def smooth(vals, fps, freq_cutoff):
    # take the points, mirror them and add them to the end to complete a loop
    # which is necessary for the fourier transform to work.
    vals_mirrored = np.array(list(vals) + list(reversed(list(vals)))[1:-1])
    N = vals_mirrored.shape[0]
    f = np.fft.fftfreq(N) * fps

    vals_mirrored_fft = np.fft.fft(vals_mirrored)
    vals_mirrored_fft[np.abs(f) > freq_cutoff] = 0
    vals_mirrored_smoothed = np.real(np.fft.ifft(vals_mirrored_fft))
    return vals_mirrored_smoothed[:vals.shape[0]]



parser = argparse.ArgumentParser()
parser.add_argument('video', type=str, help='path to video file')
parser.add_argument('--analysis', type=str, help='path to analysis file')
args = parser.parse_args()

if args.analysis is None:
    args.analysis = os.path.splitext(args.image)[0] + '.analysis'

if not os.path.exists(args.analysis):
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 107,
                           blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    # Take first frame and find corners in it

    cap = cv.VideoCapture(args.image)
    ret, old_frame = cap.read()
    frame_height, frame_width = tuple(old_frame.shape[:2])
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    frame_transforms = []

    # Create a mask image for drawing purposes

    skipped = False
    while(1):
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        if good_new.size == 0 or good_old.size == 0:
            print('points lost')
            break

        transform, inliers = cv.estimateAffinePartial2D(p0, p1)

        frame_transforms.append(transform)

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

        if p0.shape[0] < 50:
            p0 = np.unique(np.concatenate((
                p0,
                cv.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
            )), axis=0)[:100]


        # show the image
        if not skipped:
            cv.imshow('frame', resize_to_fit(img, (1000,1000)))
            k = cv.waitKey(1) & 0xff
            if k == 27:
                skipped = True

    with open(args.analysis, 'wb') as file:
        pickle.dump(np.array(frame_transforms), file)

with open(args.analysis, 'rb') as file:
    frame_transforms = pickle.load(file)
    print('frame_transforms:', frame_transforms.shape)

    num_analyzed_frames = len(frame_transforms)
    print('num_frames:', num_analyzed_frames)

    freq_cutoff = 0.2
    fps = 30

    # deltas defined through transforms
    tx = frame_transforms[:, 0, 2]
    ty = frame_transforms[:, 1, 2]
    d_theta = np.arctan2(frame_transforms[:, 1, 0], frame_transforms[:, 0, 0])
    d_s = frame_transforms[:, 0, 0] / np.cos(d_theta)

    # calculate trajectory from transforms
    px = np.cumsum(tx)
    py = np.cumsum(ty)
    theta = np.cumsum(d_theta)
    s = np.cumprod(d_s)

    # smoothed trajectory
    px_smoothed = smooth(px, fps, freq_cutoff)
    py_smoothed = smooth(py, fps, freq_cutoff)
    theta_smoothed = smooth(theta, fps, freq_cutoff)
    s_smoothed = smooth(s, fps, freq_cutoff)

    # calculate difference from real transform and the desired smoothed trajectory
    px_diff = px_smoothed - px
    py_diff = py_smoothed - py
    theta_diff = theta_smoothed - theta
    s_diff = s_smoothed - s

    # smoothed transforms
    tx_smoothed = tx + px_diff
    ty_smoothed = ty + py_diff
    d_theta_smoothed = d_theta + theta_diff
    d_s_smoothed = d_s + s_diff


    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, figsize=(10,10))
    ax1.plot(px, label='px')
    ax1.plot(px_smoothed, label='px_smoothed')
    ax1.legend()

    ax2.plot(py, label='py')
    ax2.plot(py_smoothed, label='py_smoothed')
    ax2.legend()

    ax3.plot(theta, label='theta')
    ax3.plot(theta_smoothed, label='theta_smoothed')
    ax3.legend()

    ax4.plot(s, label='s')
    ax4.plot(s_smoothed, label='s_smoothed')
    ax4.legend()
    plt.show()

    frame_transforms_smoothed = np.zeros(frame_transforms.shape)
    frame_transforms_smoothed[:, 0, 0] =  np.cos(d_theta_smoothed) # s * cos
    frame_transforms_smoothed[:, 1, 0] =  np.sin(d_theta_smoothed) # s * sin
    frame_transforms_smoothed[:, 0, 1] = -np.sin(d_theta_smoothed) # s * -sin
    frame_transforms_smoothed[:, 1, 1] = np.cos(d_theta_smoothed) # s * cos
    frame_transforms_smoothed[:, 0, 2] = tx_smoothed # tx
    frame_transforms_smoothed[:, 1, 2] = ty_smoothed # ty

    stabilized_filepath = os.path.splitext(args.image)[0] + '_stabilized.mp4'
    comparison_filepath = os.path.splitext(args.image)[0] + '_comparison.mp4'
    fourcc = cv.VideoWriter_fourcc('H','2','6','4')
    stabilized_writer = None
    comparison_writer = None

    # Read first frame to get an idea about frame size
    # and reset stream back to the start
    cap = cv.VideoCapture(args.image)
    ret, first_frame = cap.read()
    frame_height, frame_width = first_frame.shape[:2]
    frame_size = frame_width, frame_height
    is_landscape = frame_width > frame_height
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    cv.namedWindow('comparison')
    cv.moveWindow('comparison', 30, 30)
    skipped = False
    for frame_transform_smoothed in frame_transforms_smoothed:
        ret, frame = cap.read()
        if not ret:
            raise ValueError('No more frames to read. The video source might have changed.')

        stabilized = cv.warpAffine(frame, frame_transform_smoothed, frame_size)

        stabilized_resized  = resize_to_fit(stabilized, (750, 750))
        original_resized    = resize_to_fit(frame, (750, 750))
        comparison = np.hstack((original_resized, stabilized_resized))

        if stabilized_writer is None:
            stabilized_writer = cv.VideoWriter(
                stabilized_filepath,
                fourcc,
                fps,
                tuple(reversed(list(stabilized.shape[:2]))),
                True
            )
        stabilized_writer.write(stabilized)
        if comparison_writer is None:
            comparison_writer = cv.VideoWriter(
                comparison_filepath,
                fourcc,
                fps,
                tuple(reversed(list(comparison.shape[:2]))),
                True
            )
        comparison_writer.write(comparison)


        if not skipped:
            cv.imshow('comparison', comparison)
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break
                skipped = True

    cv.destroyAllWindows()
    print('\ndone')
