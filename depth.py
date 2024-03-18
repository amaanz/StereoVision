import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

disparity_map = cv.imread('disp.jpg')
cal_res = np.load('my_phone_cali.npz')
baseline = 60 #mm?

plt.imshow(disparity_map,'gray')
plt.show()


cam_mtx = cal_res['mtx']
fx = cam_mtx[0][0]

depth_map = np.zeros((disparity_map.shape[0], disparity_map.shape[1]))

count = 1
for row in disparity_map:
    print(count)
    for col in row:
        for disp in col:
            depth_map[row, col] = baseline*fx/(disparity_map[row, col, 0]+ 0.01)
    count += 1

# disparity map is now depth map
plt.imshow(depth_map,'gray')
plt.show()
