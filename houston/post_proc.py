import numpy as np
import cv2
import scipy.ndimage as ndimage
from matplotlib import pyplot as plt
import random as rng

dir = '/home/andrea/Downloads/RoadNet/DeepSegmentor-master/results/roadnet/test_latest/images/'

img_code = '1-6-3'  # straight
#img_code = '1-9-8'  # small curve
#img_code = '1-1-8'  # bad curve 

# load the images
image = cv2.imread(dir + img_code + '_image.png', cv2.IMREAD_COLOR)
label_gt = cv2.imread(dir + img_code + '_label_gt.png', cv2.IMREAD_COLOR)
label_pred = cv2.imread(dir + img_code + '_label_pred.png', cv2.IMREAD_COLOR)

# Color label_gt of red
label_gt[:,:,0] = np.zeros(label_gt.shape[0:2])
label_gt[:,:,1] = np.zeros(label_gt.shape[0:2])

# Dilate label_gt to make it more visible
kernel = np.ones((3,3), np.uint8)
thick_gt = cv2.dilate(label_gt, kernel, iterations=2, borderType=cv2.BORDER_REFLECT)

# Create the mask
mask = cv2.dilate(label_gt, kernel, iterations=20, borderType=cv2.BORDER_REFLECT)

#added_image1 = cv2.addWeighted(image, 1.0, thick_gt, 1.0, 0)
#added_image2 = cv2.addWeighted(added_image1, 0.4, mask, 0.1, 0)
#cv2.imshow('image and mask overlay', added_image2)

# Mask the predictions
masked_pred = cv2.bitwise_and(label_pred, label_pred, mask = mask[:,:,-1])
cv2.imshow('Masked Predictions', masked_pred)
cv2.imshow('Original Predictions', label_pred)

cv2.imwrite('/tmp/1_masked_preds.png', masked_pred)
cv2.imwrite('/tmp/2_original_preds.png', label_pred)


########################################################################################################################


# Thresholding the masked_pred
threshold = 100
bw = cv2.cvtColor(masked_pred, cv2.COLOR_BGR2GRAY)
_, bw = cv2.threshold(bw, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
bw_rgb = np.zeros(bw.shape+(3,), dtype=np.uint8)
bw_rgb[:,:,-1] = bw
added_image1 = cv2.addWeighted(bw_rgb, 0.8, mask, 0.4, 0)
cv2.imshow('Binary Predictions', added_image1)

"""
# Fill holes applying the closing operator
closing_kernel = np.ones((9,9), np.uint8)
closing = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, closing_kernel)
cv2.imshow('Closed inary Image', closing)
"""

# Find Contours of the thresholded masked_pred
_, contours, hierarchy = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
drawing = np.zeros(bw.shape[:2] + (3,), dtype=np.uint8)
for i in range(len(contours)):
    cv2.drawContours(drawing, contours, i, (0, 255, 0), 1, cv2.LINE_8, hierarchy, 0)

added_image1 = cv2.addWeighted(masked_pred, 0.8, mask, 0.4, 0)
added_image2 = cv2.addWeighted(added_image1, 0.4, drawing, 0.1, 0)
cv2.imshow('Edges of masked predictions', added_image2)

# Find the edges thresholding the contours
edges = cv2.bitwise_not(drawing[:,:,1])
cv2.imshow('Binary Edges', edges)

cv2.imwrite('/tmp/3_binary_edges.png', edges)

# Perform the distance transform algorithm
dist = cv2.distanceTransform(edges, cv2.DIST_L2, 3)
norm_dist = np.zeros(dist.shape)
norm_dist = cv2.normalize(dist, norm_dist, 0, 1.0, cv2.NORM_MINMAX)  # Normalize the distance image for range = {0.0, 1.0} so we can visualize and threshold it
cv2.imshow('Distance Transform Image', norm_dist)

cv2.imwrite('/tmp/4_distance_transform.png', norm_dist)

# Keep the distance values only on the centerline
centerline_3px = cv2.dilate(label_gt, kernel, iterations=1, borderType=cv2.BORDER_REFLECT)
masked_norm_dist = cv2.bitwise_and(norm_dist, norm_dist, mask=centerline_3px[:,:,-1])
cv2.imshow('/tmp/Distance transform on the centerline', masked_norm_dist)


########################################################################################################################


# Find Road Instances from the gt centerline

# Find where a road starts from the border of the image
start_points = []

for row in [0, image.shape[0]-1]: # top and bottom border
    for col in range(0, image.shape[1]):
        if(label_gt[row,col,-1]==255):
            start_points.append((row,col))

for col in [0, image.shape[1]-1]: # left and right border
    for row in range(0, image.shape[0]):
        if(label_gt[row,col,-1]==255):
            start_points.append((row,col))

temp = np.array(thick_gt)
for p in start_points:
    temp = cv2.circle(temp, (p[1],p[0]), 15, (0, 255, 0), 2)
cv2.imshow("Starting points", temp)

# DFS algorithm to find all instances
def adjacent_edges(matrix, point):
    neighbours = [
        (point[0]-1, point[1]-1),
        (point[0]-1, point[1]  ),
        (point[0]-1, point[1]+1),
        (point[0],   point[1]+1),
        (point[0]+1, point[1]+1),
        (point[0]+1, point[1]  ),
        (point[0]+1, point[1]-1),
        (point[0]  , point[1]-1)
    ]
    edges = []
    for p in neighbours:
        if 0 <= p[0] < matrix.shape[0] and 0 <= p[1] < matrix.shape[1]:
            if matrix[p] != 0:
                edges.append(p)
    
    intersection = False
    if len(edges) == 3:
        intersection = True

    return edges, intersection

discovered = np.full(label_gt.shape[:2], 0, dtype=np.uint8)
road_instance = 1

# DFS algorithm
for v in start_points:
    S = []
    S.append(v)
    while len(S) != 0:
        v = S.pop()
        edges, intersect = adjacent_edges(label_gt[:,:,-1], v)
        if intersect:
            road_instance += 1
        if discovered[v] == 0:
            discovered[v] = road_instance
            for w in edges: 
                S.append(w)

        temp = np.array(thick_gt)
        temp = cv2.circle(temp, (v[1],v[0]), 15, (0, 255, 255), 2)
        #cv2.imshow('DFS', temp)
        #cv2.waitKey(5)
    road_instance += 1

# Draw the road instances with different colors
road_instances = np.zeros(label_gt.shape, dtype=np.uint8)
for i in np.unique(discovered):
    if i > 0:
        road_instances[discovered==i] = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
cv2.imshow('Road Instances', road_instances)

cv2.imwrite('/tmp/5_road_instances.png', road_instances)

# Compute the road instaces baryceter
def baricenter(points):
    y_mean = np.mean(points[0,:])
    x_mean = np.mean(points[1,:])
    return (y_mean, x_mean)

road_instances_points = {}
baricenters = {}
for i in np.unique(discovered):
    if i > 0:
        y = []
        x = []
        for row in range(discovered.shape[0]):
            for col in range(discovered.shape[1]):
                if discovered[row,col] == i:
                    y.append(row)
                    x.append(col)
        y = np.array(y)
        x = np.array(x)
        road_instances_points[i] = np.stack([y, x])
        b = baricenter(road_instances_points[i])
        baricenters[i] = b
        temp = np.array(road_instances)

        cv2.circle(temp, (int(b[1]), int(b[0])), 10, (0, 255, 0), 2)

cv2.imshow('Baricenters', temp)

b = baricenters[1]
trans = [_ for _ in range(-20, 20)]
res = []
for dy in trans:
    for dx in trans:
        M = np.float32([1, 0, dx, 0, 1, dy]).reshape((2,3))
        rows, cols, ch = road_instances.shape
        dst = cv2.warpAffine((discovered==1).astype(np.float32), M, (cols, rows))
        cv2.imshow('Affine', dst)
        cv2.circle(dst, (int(b[1]), int(b[0])), 10, (0, 0, 255), 2)
        road_edges = drawing[:,:,1]
        cv2.imshow('Edges', road_edges)
        corr = np.sum(np.multiply(dst, road_edges))
        res.append((dy,dx,corr))

res = np.array(res)
print(res[np.argmax(res[:,-1])])

# Show the result
res_ = res[np.argmax(res[:,-1])]
dy = res_[0]
dx = res_[1]
M = np.float32([1, 0, dx, 0, 1, dy]).reshape((2,3))
rows, cols, ch = road_instances.shape
dst = cv2.warpAffine((discovered==1).astype(np.float32), M, (cols, rows))
road_edges = dst.astype(np.uint8)*255
cv2.imshow('res', road_edges)
best = cv2.addWeighted(image, 0.5, np.stack((np.zeros(road_edges.shape, dtype=np.uint8), np.zeros(road_edges.shape, dtype=np.uint8), road_edges), axis=-1), 1, 0)
cv2.imshow('Edges', best)
     
cv2.waitKey(0)

########################################################################################################################


# Plot the histogram of the distance in the centerline
masked_dist = cv2.bitwise_and(dist, dist, mask=centerline_3px[:,:,-1])
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

# Estimate the width for each road instance
estimated_road = np.zeros(label_gt.shape[:2], dtype=np.uint8)
for i in np.unique(discovered):
    if i > 0:
        inst = np.zeros(label_gt.shape[:2], dtype=np.float32)
        inst[discovered==i] = masked_dist[discovered==i]

        x = np.array([d for d in inst.flatten() if d != 0])
        unique, counts = np.unique(x, return_counts=True)
        #estimated_width = unique[np.argmax(counts)]
        estimated_width = np.mean(x)
        print(estimated_width)

        temp = np.zeros(label_gt.shape[:2], dtype=np.uint8)
        temp[discovered==i] = 255
        temp = cv2.dilate(temp, kernel, iterations=int(estimated_width), borderType=cv2.BORDER_REFLECT)

        estimated_road = cv2.bitwise_or(estimated_road, temp)

# Overlap the estimated road on the image
estimated_road_rgb = np.zeros(label_gt.shape, dtype=np.uint8)
estimated_road_rgb[:,:,-1] = estimated_road

added_image1 = cv2.addWeighted(image, 1.0, thick_gt, 1.0, 0)
added_image2 = cv2.addWeighted(added_image1, 0.8, estimated_road_rgb, 1.0, 0)
cv2.imshow('Road Estimation', added_image2)

cv2.imwrite('/tmp/6_road_estimation.png', estimated_road_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()