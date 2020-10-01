import numpy as np
import cv2
import scipy.ndimage as ndimage
from matplotlib import pyplot as plt

dir = '/home/andrea/Downloads/RoadNet/DeepSegmentor-master/results/roadnet/test_latest/images/'

#img_code = '1-6-3'  # straight
#img_code = '1-9-8'  # small curve
img_code = '1-1-8'  # bad curve 

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
plt.title("mean: " + str(mean))
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

########################################################################################################################

r = 10
y,x = np.ogrid[-r: r+1, -r: r+1]
disk = x**2+y**2 <= r**2
disk = disk.astype(float)

def test_func(values):
    delta_x = np.max(values[0]) - np.min(values[0])
    delta_y = np.max(values[1]) - np.min(values[1])
    return delta_y / delta_x

results = ndimage.generic_filter(label_gt, test_func, footprint=disk(11))


# 
inner_label_gt = np.array(label_gt[5:-5,5:-5])

print(inner_label_gt.shape)
print(label_gt.shape)

centerline_points = []
for i in range(inner_label_gt.shape[0]):
    for j in range(inner_label_gt.shape[1]):
        centerline_points.append((i,j))



# Find contours in the label_gt image
threshold = 100
label_gt_gray = cv2.cvtColor(label_gt, cv2.COLOR_BGR2GRAY)
label_gt_edges = cv2.Canny(label_gt_gray, threshold, threshold * 2)
cv2.imshow('Contours', label_gt_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

_, contours, hierarchy = cv2.findContours(label_gt_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
for i in range(len(contours)):
    drawing = np.zeros(label_gt_edges.shape[:2] + (3,), dtype=np.uint8)
    cv2.drawContours(drawing, contours, i, (0, 255, 0), 1, cv2.LINE_8, hierarchy, 0)
    cv2.imshow('Contours', drawing)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

for contour in contours:
    contour = np.squeeze(contour)
    print(contour)
    diff = np.diff(contour, axis=0)
    print(diff)
    slope = np.array([np.arctan(y/x) if x != 0 else np.pi/2 for x,y in diff])
    print(slope)

    cv2.circle(drawing, tuple(c[0][0]), 3, (255, 0, 0), 2)
    cv2.circle(drawing, tuple(c[1][0]), 3, (0, 0, 255), 2)
    cv2.imshow('Contours', drawing)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Find lines of the label_gt image
label_gt_gray = cv2.cvtColor(label_gt, cv2.COLOR_BGR2GRAY)
label_gt_edges = cv2.Canny(label_gt_gray, 50, 200)
label_gt_lines = cv2.HoughLinesP(label_gt_edges, 1, np.pi/180, 50, minLineLength=10, maxLineGap=250)
for line in label_gt_lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(label_gt, (x1, y1), (x2, y2), (0, 0, 255), 1)

cv2.imshow("Result Image", label_gt)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Load the preds

gray = cv2.cvtColor(label_pred, cv2.COLOR_BGR2GRAY)
cv2.imshow('gsadg', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Find lines from the label_pred image
gray = cv2.cvtColor(label_pred, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 200)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=10, maxLineGap=250)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(label_pred, (x1, y1), (x2, y2), (0, 0, 255), 1)

# Overlay label_gt and label_pred
added_image = cv2.addWeighted(label_pred, 0.4, label_gt, 0.1, 0)

cv2.imshow("Result Image", added_image)
cv2.waitKey(0)
cv2.destroyAllWindows()