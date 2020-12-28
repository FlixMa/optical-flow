import numpy as np
import cv2 as cv
import argparse
import math
import pickle
import matplotlib.pyplot as plt
import os
import sys
#from scipy.optimize import LinearConstraint, minimize
from scipy.interpolate import UnivariateSpline

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

def spline(vals, factor=None):
    v_min = np.min(vals)
    v_max = np.max(vals)
    v_norm = (vals - v_min) / (v_max - v_min)
    ts = np.arange(len(vals))
    spl = UnivariateSpline(ts, v_norm, s=factor)
    v_norm_smoothed = spl(ts)
    return v_norm_smoothed * (v_max - v_min) + v_min

parser = argparse.ArgumentParser()
parser.add_argument('video', type=str, help='path to video file')
parser.add_argument('--analysis', type=str, help='path to analysis file')
args = parser.parse_args()

if args.analysis is None:
    args.analysis = os.path.splitext(args.video)[0] + '.analysis'

if not os.path.exists(args.analysis):
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 17,
                           blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    # Take first frame and find corners in it

    cap = cv.VideoCapture(args.video)
    ret, old_frame = cap.read()
    frame_height, frame_width = tuple(old_frame.shape[:2])
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    frame_transforms = []

    skipped = False
    while(1):
        ret, frame = cap.read()
        if not ret:
            break

        # calculate optical flow
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
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
            cv.imshow('frame', resize_to_fit(frame, (1000,1000)))
            k = cv.waitKey(1) & 0xff
            if k == 27:
                skipped = True

    with open(args.analysis, 'wb') as file:
        pickle.dump(np.array(frame_transforms), file)

with open(args.analysis, 'rb') as file:
    frame_transforms = pickle.load(file)
    print('unpickled:', type(frame_transforms))
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

    # splined trajectory
    px_splined = spline(px, 0.7)
    py_splined = spline(py, 0.7)
    theta_splined = spline(theta, 0.7)
    s_splined = spline(s, 0.7)

    # calculate difference from real transform and the desired smoothed trajectory
    tx_splined = tx + px_splined - px
    ty_splined = ty + py_splined - py
    d_theta_splined = d_theta + theta_splined - theta
    d_s_splined = d_s + s_splined - s


    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, figsize=(10,10))
    ax1.plot(px, label='px')
    ax1.plot(px_smoothed, label='px_smoothed')
    ax1.plot(px_splined, label='px_splined')
    ax1.legend()

    ax2.plot(py, label='py')
    ax2.plot(py_smoothed, label='py_smoothed')
    ax2.plot(py_splined, label='py_splined')
    ax2.legend()

    ax3.plot(theta, label='theta')
    ax3.plot(theta_smoothed, label='theta_smoothed')
    ax3.plot(theta_splined, label='theta_splined')
    ax3.legend()

    ax4.plot(s, label='s')
    ax4.plot(s_smoothed, label='s_smoothed')
    ax4.plot(s_splined, label='s_splined')
    ax4.legend()
    plt.show()

    #sys.exit()
    s_splined = 1 # otherwise the video will be zooming in / out like crazy :-(
    frame_transforms_smoothed = np.zeros(frame_transforms.shape)
    frame_transforms_smoothed[:, 0, 0] = s_splined *  np.cos(d_theta_splined) # s *  cos
    frame_transforms_smoothed[:, 1, 0] = s_splined *  np.sin(d_theta_splined) # s *  sin
    frame_transforms_smoothed[:, 0, 1] = s_splined * -np.sin(d_theta_splined) # s * -sin
    frame_transforms_smoothed[:, 1, 1] = s_splined *  np.cos(d_theta_splined) # s *  cos
    frame_transforms_smoothed[:, 0, 2] = tx_splined # tx
    frame_transforms_smoothed[:, 1, 2] = ty_splined # ty

    '''# using fourier transform
    frame_transforms_smoothed = np.zeros(frame_transforms.shape)
    frame_transforms_smoothed[:, 0, 0] =  np.cos(d_theta_smoothed) # s * cos
    frame_transforms_smoothed[:, 1, 0] =  np.sin(d_theta_smoothed) # s * sin
    frame_transforms_smoothed[:, 0, 1] = -np.sin(d_theta_smoothed) # s * -sin
    frame_transforms_smoothed[:, 1, 1] = np.cos(d_theta_smoothed)  # s * cos
    frame_transforms_smoothed[:, 0, 2] = tx_smoothed # tx
    frame_transforms_smoothed[:, 1, 2] = ty_smoothed # ty
    '''
    stabilized_filepath = os.path.splitext(args.video)[0] + '_stabilized.mp4'
    comparison_filepath = os.path.splitext(args.video)[0] + '_comparison.mp4'
    fourcc = cv.VideoWriter_fourcc('H','2','6','4')
    stabilized_writer = None
    comparison_writer = None

    # Read first frame to get an idea about frame size
    # and reset stream back to the start
    cap = cv.VideoCapture(args.video)
    ret, first_frame = cap.read()
    frame_height, frame_width = first_frame.shape[:2]
    frame_size = frame_width, frame_height
    is_landscape = frame_width > frame_height
    cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    '''
    # determine the corners of the transformed images to get cropped size
    corners = np.array([
        [[0], [0], [1]],                        # top left
        [[0], [frame_height], [1]],             # bottom_left
        [[frame_width], [0], [1]],              # top_right
        [[frame_width], [frame_height], [1]]    # bottom_right
    ])
    transformed_corners = np.einsum('nab,mbc->nmac', frame_transforms_smoothed, corners).reshape(-1, 4, 2)
    #min_x, min_y = np.min(points.reshape(-1, 2), axis=0)
    #max_x, max_y = np.max(points.reshape(-1, 2), axis=0)
    assert num_analyzed_frames == transformed_corners.shape[0]

    # g: 0 = o + r * t
    #
    # px <= ox + rx * t     | -> px - rx * t <= ox
    # py <= oy + ry * t     | -> py - ry * t <= oy
    #
    # rows = 4 (maybe 2 only?) line constraints per point per image = 4 lines a 2 (px + py) constraints per point per image
    # columns = 8 position values (4 corners * 2 (x and y)) + 4 line parameters per image (4 lines)
    num_corner_points = 4
    num_lines = 4
    num_fixed_constraints_rows = 4
    num_rows_per_point = num_lines * 2
    num_rows_per_frame = num_rows_per_point * num_corner_points
    num_rows = num_fixed_constraints_rows + num_rows_per_frame * num_analyzed_frames
    num_columns_per_frame = num_lines
    num_columns = num_corner_points * 2 + num_columns_per_frame * num_analyzed_frames
    coefficient_matrix = np.zeros((num_rows, num_columns))
    # unconstrained bounds
    lower_bounds = np.full(num_rows, -np.inf)
    upper_bounds = np.full(num_rows, np.inf)

    o_tl = transformed_corners[:, 0]
    o_bl = transformed_corners[:, 1]
    o_tr = transformed_corners[:, 2]
    o_br = transformed_corners[:, 3]
    r_top = o_tr - o_tl
    r_bottom = o_br - o_bl
    r_left = o_bl - o_tl
    r_right = o_br - o_tr

    # make sure the corner points are in the correct order
    coefficient_matrix[:num_fixed_constraints_rows, :num_corner_points * 2] = np.array([
        [ 1, 0,   0, 0,   -1, 0,    0, 0 ], # tlx <= trx | -> tlx - trx <= 0
        [ 0, 0,   1, 0,    0, 0,   -1, 0 ], # blx <= brx | -> blx - brx <= 0
        [ 0, 1,   0,-1,    0, 0,    0, 0 ], # tly <= bly | -> tly - bly <= 0
        [ 0, 0,   0, 0,    0, 1,    0,-1 ], # try <= bry | -> try - bry <= 0
    ])
    upper_bounds[:num_fixed_constraints_rows] = 0


    for i in range(num_analyzed_frames):
        # matrix: px0 py0 px1 py1 px2 py2 px3 py3 | t0 t1 t2 t3 t4 t5 ...
        point_xy = np.array([
            [1, 0], # x (line top)
            [0, 1], # y (line top)
            [1, 0], # x (line bottom)
            [0, 1], # y (line bottom)
            [1, 0], # x (line left)
            [0, 1], # y (line left)
            [1, 0], # x (line right)
            [0, 1], # y (line right)
        ])
        point_rxy = np.array([
            [-r_top[i, 0],   0,               0,              0               ],
            [-r_top[i, 1],   0,               0,              0               ],
            [ 0,            -r_bottom[i, 0],  0,              0               ],
            [ 0,            -r_bottom[i, 1],  0,              0               ],
            [ 0,             0,              -r_left[i, 0],   0               ],
            [ 0,             0,              -r_left[i, 1],   0               ],
            [ 0,             0,               0,             -r_right[i, 0]   ],
            [ 0,             0,               0,             -r_right[i, 1]   ]
        ])

        point_lb = np.array([
            -np.inf, -np.inf,
            o_bl[i, 0], o_bl[i, 1],
            o_tl[i, 0], o_tl[i, 1],
            -np.inf, -np.inf,
        ])
        point_ub = np.array([
            o_tl[i, 0], o_tl[i, 1],
            np.inf, np.inf,
            np.inf, np.inf,
            o_tr[i, 0], o_tr[i, 1],
        ])

        for p in range(num_corner_points):
            start_row = num_fixed_constraints_rows + i * num_rows_per_frame + p * num_rows_per_point
            end_row = num_fixed_constraints_rows + i * num_rows_per_frame + (p+1) * num_rows_per_point

            start_column_rxy = num_corner_points * 2 + i * num_columns_per_frame
            end_column_rxy = num_corner_points * 2 + (i+1) * num_columns_per_frame

            start_column_xy = p * 2
            end_column_xy = (p+1) * 2

            coefficient_matrix[start_row:end_row, start_column_xy:end_column_xy] = point_xy
            coefficient_matrix[start_row:end_row, start_column_rxy:end_column_rxy] = point_rxy

            lower_bounds[start_row:end_row] = point_lb
            upper_bounds[start_row:end_row] = point_ub

    print(coefficient_matrix.shape)
    constraints = LinearConstraint(coefficient_matrix, lower_bounds, upper_bounds)

    def inverse_area(x, *args, **kwargs):
        p_tl = x[:2]
        p_bl = x[2:4]
        p_tr = x[4:6]

        v_tl_bl = p_bl - p_tl
        v_tl_tr = p_tr - p_tl

        # area = |v_tl_bl| * |v_tl_tr| = sqrt(x1*x1+y1*y1) * sqrt(x2*x2+y2*y2)
        # since we dont want the actual area but only a measure of how big it is
        # we can leave the sqrt
        a_squared = v_tl_bl.dot(v_tl_bl) * v_tl_tr.dot(v_tl_tr)

        # since we would like to maximize the area, but (using scipy) only
        # minimizing is possible -> we need to minimize the negative area
        return -a_squared


    initial_guess = np.zeros(num_columns) # populate with corners



    sys.exit()

    print(minimize(inverse_area, initial_guess, method='trust-constr', constraints=constraints))
    '''
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
                skipped = True
        else:
            print('.', end='')
            sys.stdout.flush()

    cv.destroyAllWindows()
    print('\ndone')
