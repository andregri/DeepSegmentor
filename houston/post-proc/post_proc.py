import numpy as np
import cv2
import scipy.ndimage as ndimage
from matplotlib import pyplot as plt

from visual_utils import *
from post_proc_utils import *

dir = '/home/andrea/Downloads/RoadNet/DeepSegmentor-master/results/roadnet/test_latest/images/'

#img_code = '1-6-3'  # straight
img_code = '1-9-8'  # small curve
#img_code = '1-1-8'  # bad curve 

# load the images
image_bgr = cv2.imread(dir + img_code + '_image.png', cv2.IMREAD_COLOR)
label_gt_bgr = cv2.imread(dir + img_code + '_label_gt.png', cv2.IMREAD_COLOR)
label_pred_bgr = cv2.imread(dir + img_code + '_label_pred.png', cv2.IMREAD_COLOR)

# Dilate label_gt to make it more visible
thick_gt_bgr = thick(label_gt_bgr)

# Create the mask
kernel = np.ones((3,3), np.uint8)
mask_bgr = cv2.dilate(label_gt_bgr, kernel, iterations=20, borderType=cv2.BORDER_REFLECT)

overlaid_bgr = overlayImages([image_bgr, green(thick_gt_bgr), red(mask_bgr)],[1.0, 1.0, 0.4])
cv2.imshow('image and mask overlay', overlaid_bgr)

# Mask the predictions
masked_pred_bgr = cv2.bitwise_and(label_pred_bgr, label_pred_bgr, mask = mask_bgr[:,:,0])
cv2.imshow('Masked Predictions', masked_pred_bgr)
cv2.imshow('Original Predictions', label_pred_bgr)

cv2.imwrite('/tmp/1_masked_preds.png', masked_pred_bgr)
cv2.imwrite('/tmp/2_original_preds.png', label_pred_bgr)


########################################################################################################################


# Thresholding the masked_pred
threshold = 100
masked_pred_gray = cv2.cvtColor(masked_pred_bgr, cv2.COLOR_BGR2GRAY)
_, binary_masked_pred_gray = cv2.threshold(masked_pred_gray, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
overlaid_bgr = overlayImages([red(mask_bgr), grayToBGR(binary_masked_pred_gray)], [0.4, 1.0])
cv2.imshow('Binary Predictions', overlaid_bgr)

"""
# Fill holes applying the closing operator
closing_kernel = np.ones((9,9), np.uint8)
closing = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, closing_kernel)
cv2.imshow('Closed inary Image', closing)
"""

# Find Contours of the thresholded masked_pred
_, contours, hierarchy = cv2.findContours(binary_masked_pred_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
pred_contours_bgr = np.zeros_like(masked_pred_bgr)
for i in range(len(contours)):
    cv2.drawContours(pred_contours_bgr, contours, i, (0, 255, 0), 1, cv2.LINE_8, hierarchy, 0)

overlaid_bgr = overlayImages([mask_bgr, masked_pred_bgr, pred_contours_bgr], [0.2, 1.0, 1.0])
cv2.imshow('Edges of masked predictions', overlaid_bgr)


# Find the edges thresholding the contours
binary_edges_gray = cv2.bitwise_not(pred_contours_bgr[:,:,1])
cv2.imshow('Binary Edges', binary_edges_gray)

cv2.imwrite('/tmp/3_binary_edges.png', binary_edges_gray)


# Perform the distance transform algorithm
distance_transform_gray = cv2.distanceTransform(binary_edges_gray, cv2.DIST_L2, 3)
norm_distance_transform_gray = np.zeros_like(distance_transform_gray)
norm_distance_transform_gray = cv2.normalize(distance_transform_gray, norm_distance_transform_gray, 0, 1.0, cv2.NORM_MINMAX)  # Normalize the distance image for range = {0.0, 1.0} so we can visualize and threshold it
cv2.imshow('Distance Transform Image', norm_distance_transform_gray)

cv2.imwrite('/tmp/4_distance_transform.png', norm_distance_transform_gray)


# Keep the distance values only on the centerline
masked_norm_dist_gray = cv2.bitwise_and(norm_distance_transform_gray, norm_distance_transform_gray, mask=thick(label_gt_bgr,3)[:,:,0])
cv2.imshow('/tmp/Distance transform on the centerline', masked_norm_dist_gray)


# Plot the histogram of the distance in the centerline
kernel = np.ones((3,3), np.uint8)
centerline_3px = cv2.dilate(label_gt_bgr, kernel, iterations=1, borderType=cv2.BORDER_REFLECT)
masked_dist = cv2.bitwise_and(distance_transform_gray, distance_transform_gray, mask=centerline_3px[:,:,-1])

x = np.array([d for d in masked_dist.flatten() if d != 0])
bins = 25
bin_width = (x.max()-0.1)/bins
plt.hist(x, bins=bins, range=(0.1, x.max()), rwidth=0.9*bin_width)
ticks = [0.1 + bin_width * i for i in range(bins)]
plt.xticks(ticks=ticks, rotation=70)
mean = np.mean(x)
var  = np.std(x)
plt.title("mean: " + str(mean) + "  std: " + str(var))
#plt.show()


########################################################################################################################


# Find Road Instances from the gt centerline


# Find where a road starts from the border of the image
start_points = findRoadStartingPoints(label_gt_bgr[:,:,-1])
cv2.imshow("Starting points", addCircles(red(thick_gt_bgr), start_points))


# DFS algorithm to find all instances
discovered_gray = DFS(label_gt_bgr[:,:,0], start_points)


# Draw the road instances with different colors
road_instances_bgr = colorInstances(discovered_gray, np.unique(discovered_gray))
cv2.imshow('Road Instances', road_instances_bgr)


cv2.imwrite('/tmp/5_road_instances.png', road_instances_bgr)


road_instances_points = {}
baricenters = {}
for i in np.unique(discovered_gray):
    if i > 0:
        y = []
        x = []
        for row in range(discovered_gray.shape[0]):
            for col in range(discovered_gray.shape[1]):
                if discovered_gray[row,col] == i:
                    y.append(row)
                    x.append(col)
        y = np.array(y)
        x = np.array(x)
        road_instances_points[i] = np.stack([y, x])
        b = intBaricenter(road_instances_points[i])
        baricenters[i] = b

cv2.imshow('Baricenters', addCircles(road_instances_bgr, list(baricenters.values())))


b = baricenters[1]

translation_region = [_ for _ in range(-20, 20)]
correlations = []
for dy in translation_region:
    for dx in translation_region:
        M = np.float32([1, 0, dx, 0, 1, dy]).reshape((2,3))
        rows, cols, ch = road_instances_bgr.shape
        translated_gray = cv2.warpAffine((discovered_gray==1).astype(np.float32), M, (cols, rows))
        white_binary_edges_gray = cv2.bitwise_not(binary_edges_gray)
        corr = np.sum(np.multiply(translated_gray, white_binary_edges_gray))
        correlations.append((dy,dx,corr))

res = np.array(correlations)
print(res[np.argmax(res[:,-1])])

# Show the result
res_ = res[np.argmax(res[:,-1])]
dy = res_[0]
dx = res_[1]
M = np.float32([1, 0, dx, 0, 1, dy]).reshape((2,3))
rows, cols, ch = road_instances_bgr.shape
dst = cv2.warpAffine((discovered_gray==1).astype(np.float32), M, (cols, rows))
binary_edges_gray = dst.astype(np.uint8)*255

cv2.imshow('Road Edges', overlayImages([image_bgr, grayToBGR(binary_edges_gray, 0, 0, 1)],[0.5, 1]))


########################################################################################################################


# Estimate the width for each road instance
estimated_road_gray = np.zeros(label_gt_bgr.shape[:2], dtype=np.uint8)
for i in np.unique(discovered_gray):
    if i > 0:
        inst = np.zeros(label_gt_bgr.shape[:2], dtype=np.float32)
        inst[discovered_gray==i] = masked_dist[discovered_gray==i]

        x = np.array([d for d in inst.flatten() if d != 0])
        unique, counts = np.unique(x, return_counts=True)
        #estimated_width = unique[np.argmax(counts)]
        estimated_width = np.mean(x)
        print(estimated_width)

        temp = np.zeros(label_gt_bgr.shape[:2], dtype=np.uint8)
        temp[discovered_gray==i] = 255
        temp = cv2.dilate(temp, kernel, iterations=int(estimated_width), borderType=cv2.BORDER_REFLECT)

        estimated_road_gray = cv2.bitwise_or(estimated_road_gray, temp)

# Overlap the estimated road on the image
overlaid_bgr = overlayImages([image_bgr, thick_gt_bgr, grayToBGR(estimated_road_gray, 0, 0, 1)],[1.0, 1.0, 0.8])
cv2.imshow('Road Estimation', overlaid_bgr)

cv2.waitKey(0)
cv2.destroyAllWindows()