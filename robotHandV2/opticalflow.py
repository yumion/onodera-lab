import numpy as np
import cv2
import time
from utils.realsensecv import RealsenseCapture
cap = RealsenseCapture()
cap.start()
time.sleep(5)

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))


# Take first frame and find corners in it
ret, frames = cap.read()
old_frame = frames[0]
depth_image = np.array(cap.depth_frame.get_data())

#   距離[m] = depth * depth_scale
depth_sensor = cap.profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
clipping_distance_in_meters = 0.3  # [meter] 何メートル以内をくり抜くか
clipping_distance = clipping_distance_in_meters / depth_scale  # depth = distance / depth_scale

# Depth画像前処理(1m以内を画像化)
grey_color = 0
depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
bg_removed = np.where((depth_image_3d > clipping_distance) | (
    depth_image_3d <= 0), grey_color, old_frame)

old_gray = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
# print('start: ', p0)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while True:
    ret, frames = cap.read()
    frame = frames[0]
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        # print('pre: ', good_old)
        # print('next: ', good_new)
    else:
        break

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    img = cv2.add(frame, mask)

    cv2.imshow('frame', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()
