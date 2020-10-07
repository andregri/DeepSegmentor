import numpy as np
import cv2
import seaborn as sns

def grayToBGR(grayscale_image, b=1, g=1, r=1):
    assert 0 <= b <= 1.0
    assert 0 <= g <= 1.0
    assert 0 <= r <= 1.0
    bgr_image = np.zeros(grayscale_image.shape+(3,), dtype=grayscale_image.dtype)

    bgr_image[:,:,0] = b * grayscale_image
    bgr_image[:,:,1] = g * grayscale_image
    bgr_image[:,:,2] = r * grayscale_image

    return bgr_image

def thick(input_image, stroke=3):
    kernel = np.ones((stroke,stroke), np.uint8)
    output_image = cv2.dilate(input_image, kernel, iterations=1, borderType=cv2.BORDER_REFLECT)
    return output_image

def tuneBGR(input_image, b, g, r):
    assert 0 <= b <= 1.0
    assert 0 <= g <= 1.0
    assert 0 <= r <= 1.0

    input_grayscale = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    output_image = np.zeros(input_image.shape, dtype=input_image.dtype)
    for i, ch in zip(range(0,3), [b, g, r]):
        output_image[:,:,i] = ch * input_grayscale
    return output_image

def red(input_image):
    return tuneBGR(input_image, 0, 0, 1)

def green(input_image):
    return tuneBGR(input_image, 0, 1, 0)

def overlayImages(input_images, weights):
    output_image = np.zeros_like(input_images[0])
    for image, w in zip(input_images, weights):
        output_image = cv2.addWeighted(output_image, 1.0, image, w, 0)
    return output_image

def addCircles(input_image, point_coords, radius=15, color=(0, 255, 0), thickness=2):
    output_image = np.array(input_image)
    for p in point_coords:
        output_image = cv2.circle(output_image, (p[1],p[0]), radius, color, thickness)
    return output_image

def colorInstances(input_image, labels_to_color):
    assert len(input_image.shape) == 2

    output_image = np.zeros(input_image.shape + (3,), dtype=np.uint8)
    colors = sns.color_palette("hls", len(labels_to_color))
    for c, label in enumerate(labels_to_color):
        if label != 0:
            output_image[input_image==label] = (colors[c][2]*255, colors[c][1]*255, colors[c][0]*255)
    return output_image
