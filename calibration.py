import numpy as np
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt

#rember to read files with /

def calibration(chess_photos_dir, chess_x_y_grid = (7, 6)):

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    rows, cols = chess_x_y_grid
    objp = np.zeros((rows*cols,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:cols].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    #getting the image list
    images = glob.glob(f"{chess_photos_dir}/*.jpg")
    # print(images )
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cv.imshow("THIS", gray)
        cv.waitKey(0)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (rows,cols), None)
        print(ret)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
            # Draw and display the corners
            cv.drawChessboardCorners(img, (rows,cols), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(20)
    cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    print(f"total error: {mean_error / len(objpoints)}")

    return (ret, mtx, dist, rvecs, tvecs)

def rem_distortion(image, mtx, dist, remapping = False):
    rows, cols = image.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (cols, rows),
                                                     1, (cols, rows))

    # undistort
    if not remapping:
        dst = cv.undistort(image, mtx, dist, None, newcameramtx)
    else:
        mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (cols, rows), 5)
        dst = cv.remap(image, mapx, mapy, cv.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]

    return dst


test_image = cv.imread("cal_test.jpg")
cal_res = calibration("my_phone_chess", (7, 6))
unditorted_img = rem_distortion(test_image, cal_res[1], cal_res[2], False )
np.savez('my_phone_cali.npz',
        ret=cal_res[0], mtx=cal_res[1], dist=cal_res[2], rvecs=cal_res[3], tvecs=cal_res[4])

cv.imshow("undist", unditorted_img)
cv.waitKey(0)
