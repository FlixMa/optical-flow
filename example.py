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

parser = argparse.ArgumentParser(description='This sample demonstrates Lucas-Kanade Optical Flow calculation. \
                                              The example file can be downloaded from: \
                                              https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
parser.add_argument('image', type=str, help='path to image file')
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


    cv.namedWindow("frame")
    cv.moveWindow("frame", 20, 20);

    min_diff_x = 0
    min_diff_y = 0
    max_diff_x = 0
    max_diff_y = 0

    current_loc_x = 0
    current_loc_y = 0

    frame_locations = []

    # Create a mask image for drawing purposes
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

        diffsX = []
        diffsY = []

        # draw the tracks
        mask = np.zeros_like(old_frame)
        for i, (new,old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            diffsX.append(c - a)
            diffsY.append(d - b)

            mask = cv.line(mask, (a,b), (c,d), color[i].tolist(), 2)
            mask = cv.circle(mask, (a,b), 3, color[i].tolist(), -1)

        diffX = np.median(diffsX)
        diffY = np.median(diffsY)


        current_loc_x += diffX
        current_loc_y += diffY

        frame_locations.append((current_loc_x, current_loc_y))

        min_diff_x = min(current_loc_x, min_diff_x)
        min_diff_y = min(current_loc_y, min_diff_y)
        max_diff_x = max(current_loc_x, max_diff_x)
        max_diff_y = max(current_loc_y, max_diff_y)


        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

        if p0.shape[0] < 50:
            p0 = np.unique(np.concatenate((
                p0,
                cv.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
            )), axis=0)[:100]

        # show the image
        img = cv.add(frame, mask)
        img = cv.line(img, (20,20), (20+int(diffX*5),20+int(diffY*5)), (0, 255, 0), 3)
        cv.imshow('frame', resize_to_fit(img, (1000,1000)))

        k = cv.waitKey(1) & 0xff
        if k == 27:
            break


    min_diff_x = math.floor(min_diff_x)
    min_diff_y = math.floor(min_diff_y)
    max_diff_x = math.ceil(max_diff_x)
    max_diff_y = math.ceil(max_diff_y)

    sys.exit()

    with open(args.analysis, 'wb') as file:
        pickle.dump((
            (frame_height, frame_width),
            frame_locations,
            min_diff_x,
            min_diff_y,
            max_diff_x,
            max_diff_y
        ), file)



with open(args.analysis, 'rb') as file:
    frame_size, frame_locations, min_diff_x, min_diff_y, max_diff_x, max_diff_y = pickle.load(file)
    frame_height, frame_width = frame_size
    canvas_height = frame_height + max_diff_y - min_diff_y
    canvas_width = frame_width + max_diff_x - min_diff_x

    num_analyzed_frames = len(frame_locations)
    print('num_frames:', num_analyzed_frames)
    print(min_diff_x, min_diff_y, max_diff_x, max_diff_y)

    xs = list(map(lambda frame: frame[0], frame_locations))
    ys = list(map(lambda frame: frame[1], frame_locations))

    # take the points, mirror them and add them to the end to complete a loop
    # which is necessary for the fourier transform to work.
    xs = np.array(xs + list(reversed(xs))[1:-1])
    ys = np.array(ys + list(reversed(ys))[1:-1])

    N = num_analyzed_frames * 2 - 2
    samples_per_second = 30
    f = np.fft.fftfreq(N) * samples_per_second

    freq_cutoff = 0.2
    x_fft = np.fft.fft(xs)
    x_fft[np.abs(f) > freq_cutoff] = 0
    xs_smoothed = np.real(np.fft.ifft(x_fft))

    y_fft = np.fft.fft(ys)
    y_fft[np.abs(f) > freq_cutoff] = 0
    ys_smoothed = np.real(np.fft.ifft(y_fft))

    xs = xs[:num_analyzed_frames]
    ys = ys[:num_analyzed_frames]
    xs_smoothed = xs_smoothed[:num_analyzed_frames]
    ys_smoothed = ys_smoothed[:num_analyzed_frames]

    xs_diff = xs_smoothed - xs
    ys_diff = ys_smoothed - ys

    min_diff_x = math.floor(np.min(xs_diff))
    min_diff_y = math.floor(np.min(ys_diff))
    max_diff_x = math.ceil(np.max(xs_diff))
    max_diff_y = math.ceil(np.max(ys_diff))

    print(min_diff_x, min_diff_y, max_diff_x, max_diff_y)

    x_locations = np.clip(0, xs - min_diff_x, canvas_width - frame_width)
    y_locations = np.clip(0, xs - min_diff_y, canvas_height - frame_height)
    x_locations_smoothed = np.clip(0, xs_smoothed - min_diff_x, canvas_width - frame_width)
    y_locations_smoothed = np.clip(0, ys_smoothed - min_diff_y, canvas_height - frame_height)

    '''
    last_location_x = None
    last_location_y = None
    last_location_smoothed_x = None
    last_location_smoothed_y = None

    skipped = False

    frame_infos = []
    line_canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    for x, y, x_smoothed, y_smoothed in list(zip(xs, ys, xs_smoothed, ys_smoothed)):
        #new_location_x = max(0, min(int(x - min_diff_x), canvas_width - frame_width))
        #new_location_y = max(0, min(int(y - min_diff_y), canvas_height - frame_height))
        #new_location_smoothed_x = max(0, min(int(x_smoothed - min_diff_x), canvas_width - frame_width))
        #new_location_smoothed_y = max(0, min(int(y_smoothed - min_diff_y), canvas_height - frame_height))

        if last_location_x is not None and last_location_y is not None and\
            last_location_x is not None and last_location_y is not None:

            #last_center_smoothed = (last_location_smoothed_x + frame_width//2, last_location_smoothed_y + frame_height//2)
            #new_center_smoothed = (new_location_smoothed_x + frame_width//2, new_location_smoothed_y + frame_height//2)

            #last_center = (last_location_x + frame_width//2, last_location_y + frame_height//2)
            #new_center = (new_location_x + frame_width//2, new_location_y + frame_height//2)

            #line_canvas = cv.line(line_canvas, new_center_smoothed, new_center, (0, 255, 255), 4)
            #line_canvas = cv.line(line_canvas, last_center, new_center, (0, 255, 0), 4)
            #line_canvas = cv.line(line_canvas, last_center_smoothed, new_center_smoothed, (0, 0, 255), 4)

            line_canvas = cv.line(
                line_canvas,
                (new_location_smoothed_x, new_location_smoothed_y),
                (new_location_x, new_location_y),
                (0, 255, 255),
                4
            )
            line_canvas = cv.line(
                line_canvas,
                (last_location_x, last_location_y),
                (new_location_x, new_location_y),
                (0, 255, 0),
                4
            )
            line_canvas = cv.line(
                line_canvas,
                (last_location_smoothed_x, last_location_smoothed_y),
                (new_location_smoothed_x, new_location_smoothed_y),
                (0, 0, 255),
                4
            )

        #last_location_x = new_location_x
        #last_location_y = new_location_y
        #last_location_smoothed_x = new_location_smoothed_x
        #last_location_smoothed_y = new_location_smoothed_y


    cv.imshow('frame', cv.resize(line_canvas, (0,0), fx=0.2, fy=0.2))
    cv.imwrite('out.png', line_canvas)
    '''

    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(xs, label='xs')
    ax.plot(ys, label='ys')
    ax.plot(xs_smoothed, label='xs_smoothed')
    ax.plot(ys_smoothed, label='ys_smoothed')
    ax.legend()
    plt.show()

    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(xs, ys, label='orig')
    ax.plot(xs_smoothed, ys_smoothed, label='smoothed')
    ax.legend()
    plt.show()



    skipped = False
    cap = cv.VideoCapture(args.image)
    ret, first_frame = cap.read()
    original_resized = resize_to_fit(first_frame, (1000, 1000))

    # calculate output frame sizes
    landscape = frame_width > frame_height
    stabilized_width = frame_width - abs(min_diff_x) - abs(max_diff_x)
    stabilized_height = frame_height - abs(min_diff_y) - abs(max_diff_y)
    stabilized_size = (stabilized_width, stabilized_height)
    print('stabilized_size:', stabilized_size)

    stabilized_filepath = os.path.splitext(args.image)[0] + '_stabilized.mp4'
    comparison_filepath = os.path.splitext(args.image)[0] + '_comparison.mp4'
    fourcc = cv.VideoWriter_fourcc('H','2','6','4')
    stabilized_writer = None
    comparison_writer = None


    cv.namedWindow('comparison')
    cv.moveWindow('comparison', 30, 30)
    for diff_x, diff_y in zip(xs_diff, ys_diff):
        ret, frame = cap.read()
        if not ret:
            raise ValueError('No more frames to read. The video source might have changed.')
        print('.', end='')
        sys.stdout.flush()

        top    = abs(min_diff_y) + int(diff_y)
        bottom = frame_height - abs(max_diff_y) + int(diff_y)
        left   = abs(min_diff_x) + int(diff_x)
        right  = frame_width - abs(max_diff_x) + int(diff_x)
        stabilized = frame[top:bottom, left:right].copy()

        frame = cv.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 4)

        stabilized_resized  = resize_to_fit(stabilized, (1000, 1000))
        original_resized    = resize_to_fit(frame, (1000, 1000))
        comparison = np.vstack((original_resized, stabilized_resized)) if landscape else np.hstack((original_resized, stabilized_resized))

        if stabilized_writer is None:
            stabilized_writer = cv.VideoWriter(
                stabilized_filepath,
                fourcc,
                samples_per_second,
                tuple(reversed(list(stabilized.shape[:2]))),
                True
            )
        stabilized_writer.write(stabilized)
        if comparison_writer is None:
            comparison_writer = cv.VideoWriter(
                comparison_filepath,
                fourcc,
                samples_per_second,
                tuple(reversed(list(comparison.shape[:2]))),
                True
            )
        comparison_writer.write(comparison)


        if not skipped:
            cv.imshow('comparison', comparison)
            k = cv.waitKey(30) & 0xff
            if k == 27:
                skipped = True

    cv.destroyAllWindows()
    print('\ndone')
