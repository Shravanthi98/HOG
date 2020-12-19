#!/usr/bin/env python3
# ECE 471/536: Assignment 3 submission template


#Using "as" nicknames a library so you don't have to use the full name
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse as ap
import pprint

pp = pprint.PrettyPrinter(indent=4)

#Prevents python3 from failing
TODO = None

#Epsilon 
EPS = 1e-6

"""Extract Histogram of Gradient features

Parameters
----------
X : ndarray NxHxW array where N is the number of instances/images 
                              HxW is the image dimensions

Returns
    features : NxD narray containing the histogram features (D = 2304)(10000x144) 
-------

"""
#20 marks: Histogram of Gradients


# Reference for the algorithm: Navneet Dalal and Bill Triggs, "Histogram of Oriented Gradients for Human Detection"
def hog(X):
    # lists to store the magnitude and angle of the gradient
    grad_mag1 = []
    grad_angle1 = []
    # N = number of images
    N = len(X)
    # Gradient computation for all the images
    for i in range(0, N):
        grad_x = cv2.Sobel(X[i], cv2.CV_32F, 1, 0, 1)
        grad_y = cv2.Sobel(X[i], cv2.CV_32F, 0, 1, 1)
        grad_m, grad_ang = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
        grad_ang = grad_ang % 180
        grad_mag1.append(grad_m)
        grad_angle1.append(grad_ang)
    grad_mag = np.stack(grad_mag1)
    grad_angle = np.stack(grad_angle1)
    # Split the magnitude and angle into 8 x 8 cells
    # split_mag, split_angle = N X 16 X 64
    split_mag = split_into_cells(grad_mag, 8)
    split_angle = split_into_cells(grad_angle, 8)
    # bins of the histogram
    histogram_y = np.array([0, 20, 40, 60, 80, 100, 120, 140, 160])
    n, r, c = split_angle.shape
    sum1 = 0
    sum2 = 0
    s1 = 0
    s2 = 0
    final_hist = []
    # Histogram computation for every cell in every image
    for img in range(0, n):
        hist_list = []
        for block in range(0, r):
            mag_array = split_mag[img][block]
            ang_array = split_angle[img][block]
            histogram_x = np.zeros(9, dtype="float32")
            for cell in range(0, c):
                mag = mag_array[cell]
                ang = ang_array[cell]
                # If angle value directly falls into one of the bins i.e. 100%
                if ang % 20 == 0:
                    ind = int(ang / 20)
                    if ind == 9:
                        if histogram_x[0] != 0:
                            s1 = histogram_x[0]
                            histogram_x[0] = s1 + mag
                        else:
                            histogram_x[0] = mag
                    else:
                        if histogram_x[ind] != 0:
                            s2 = histogram_x[ind]
                            histogram_x[ind] = s2 + mag
                        else:
                            histogram_x[ind] = mag
                # When the magnitude gets divided between two bins
                else:
                    diff1 = np.abs(ang - histogram_y[0])
                    idx1 = 0
                    # Comparing the angle value to all the bin values and considering two bins with lowest difference
                    for i in range(1, 9):
                        cdiff = np.abs(ang - histogram_y[i])
                        if cdiff < diff1:
                            idx1 = i
                            diff1 = cdiff
                    # When bin index = 8 i.e. angle = 180, maps to 0
                    if idx1 == 8:
                        idxn = 0
                        d1 = np.abs(ang - 180)
                    else:
                        idxn = idx1 + 1
                        d1 = np.abs(ang - histogram_y[idxn])
                    d2 = np.abs(ang - histogram_y[idx1 - 1])
                    if d1 < d2:
                        idx2 = idxn
                        diff2 = d1
                    else:
                        idx2 = idx1 - 1
                        diff2 = d2
                    # magnitude weights for each of the two bins based on the ratio
                    weight1 = ((diff2 / 20) * mag)
                    weight2 = ((diff1 / 20) * mag)
                    if histogram_x[idx1] != 0:
                        sum1 = histogram_x[idx1]
                        histogram_x[idx1] = sum1 + weight1
                    else:
                        histogram_x[idx1] = weight1
                    if histogram_x[idx2] != 0:
                        sum2 = histogram_x[idx2]
                        histogram_x[idx2] = sum2 + weight2
                    else:
                        histogram_x[idx2] = weight2
            hist_list.append(histogram_x)
        hist_concatenated = np.vstack(hist_list)
        hist1_concatenated = hist_concatenated.flatten()
        final_hist.append(hist1_concatenated)
    # final hist = array of shape N X 144, feature vector = 144, N = number of images
    final_hist = np.vstack(final_hist)
    return final_hist
    pass


# 5 marks: Split the input matrix A into cells
def split_into_cells(A, cell_size=8):
    final_split = []
    # Split an image into 8 x 8 cells for all N images
    for index in range(0, len(A)):
        image = A[index]
        cell = []
        i = 0
        # An image to 16 blocks of 8 x 8 cells
        for row in range(0, int(cell_size / 2)):
            j = 0
            for col in range(0, int(cell_size / 2)):
                single_cell = image[i: i + cell_size, j: j + cell_size]
                # Flatten 8 x 8 into 64
                flattened_single_cell = single_cell.flatten()
                cell.append(flattened_single_cell)
                j = j + cell_size
            i = i + cell_size
        stacked_cell = np.stack(cell)
        final_split.append(stacked_cell)
    # final_split_array = N X 16 X 64. N images of 16 blocks/image with 64-length cells/block.
    final_split_array = np.stack(final_split)
    return final_split_array
    pass

