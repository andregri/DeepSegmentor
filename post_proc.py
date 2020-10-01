import numpy as np
import cv2
import scipy.ndimage as ndimage
from matplotlib import pyplot as plt

dir = '/home/andrea/Downloads/RoadNet/DeepSegmentor-master/results/roadnet/test_latest/images/'

#img_code = '1-6-3'  # straight
img_code = '1-9-8'  # small curve
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
#cv2.imshow('mask overlay', mask)

added_image1 = cv2.addWeighted(image, 1.0, thick_gt, 1.0, 0)
added_image2 = cv2.addWeighted(added_image1, 0.4, mask, 0.1, 0)
#cv2.imshow('image and mask overlay', added_image2)

# Mask the predictions
masked_pred = cv2.bitwise_and(label_pred, label_pred, mask = mask[:,:,-1])
cv2.imshow('masked', masked_pred)
cv2.imshow('original', label_pred)


########################################################################################################################


# Thresholding the masked_pred
threshold = 100
bw = cv2.cvtColor(masked_pred, cv2.COLOR_BGR2GRAY)
_, bw = cv2.threshold(bw, 40, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
bw_rgb = np.zeros(bw.shape+(3,), dtype=np.uint8)
bw_rgb[:,:,-1] = bw
added_image1 = cv2.addWeighted(bw_rgb, 0.8, mask, 0.4, 0)
cv2.imshow('Binary Image', added_image1)

# Find Contours of the thresholded masked_pred
_, contours, hierarchy = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
drawing = np.zeros(bw.shape[:2] + (3,), dtype=np.uint8)
for i in range(len(contours)):
    cv2.drawContours(drawing, contours, i, (0, 255, 0), 1, cv2.LINE_8, hierarchy, 0)

added_image1 = cv2.addWeighted(masked_pred, 0.8, mask, 0.4, 0)
added_image2 = cv2.addWeighted(added_image1, 0.4, drawing, 0.1, 0)
cv2.imshow('masked edges', added_image2)

# Find the edges thresholding the contours
edges = cv2.bitwise_not(drawing[:,:,1])
cv2.imshow('edges', edges)

# Perform the distance transform algorithm
dist = cv2.distanceTransform(edges, cv2.DIST_L2, 3)
norm_dist = np.zeros(dist.shape)
norm_dist = cv2.normalize(dist, norm_dist, 0, 1.0, cv2.NORM_MINMAX)  # Normalize the distance image for range = {0.0, 1.0} so we can visualize and threshold it
cv2.imshow('Distance Transform Image', norm_dist)

# Keep the distance values only on the centerline
centerline_3px = cv2.dilate(label_gt, kernel, iterations=1, borderType=cv2.BORDER_REFLECT)
masked_norm_dist = cv2.bitwise_and(norm_dist, norm_dist, mask=centerline_3px[:,:,-1])
cv2.imshow('masked_dist', masked_norm_dist)

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
plt.show()

#cv2.waitKey(0)
cv2.destroyAllWindows()

########################################################################################################################

# Create the mask
unique, counts = np.unique(x, return_counts=True)
estimated_width = mean
print("estimated width: " + str(estimated_width))
estimated_road = cv2.dilate(label_gt, kernel, iterations=estimated_width, borderType=cv2.BORDER_REFLECT)

added_image1 = cv2.addWeighted(image, 1.0, thick_gt, 1.0, 0)
added_image2 = cv2.addWeighted(added_image1, 0.8, estimated_road, 1.0, 0)
cv2.imshow('Estimation', added_image2)

cv2.waitKey(0)
cv2.destroyAllWindows()