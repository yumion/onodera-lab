import cv2


def calc_center(img):
    '''重心座標(x,y)を求める'''
    mu = cv2.moments(img, False)
    x = int(mu["m10"] / (mu["m00"] + 1e-7))
    y = int(mu["m01"] / (mu["m00"] + 1e-7))
    return x, y
