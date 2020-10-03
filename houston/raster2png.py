import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import cv2
import matplotlib

ds = gdal.Open("/home/andrea/Downloads/Final RGB HR Imagery/UH_NAD83_271460_3289689_centerline.tif")
myarray = np.array(ds.GetRasterBand(1).ReadAsArray(), dtype=np.uint8)
myarray[myarray==1] = 255
plt.imshow(myarray, cmap='gray')
plt.show()

rgb = np.full(myarray.shape + (3,), 255, dtype=np.uint8)
rgb[myarray==255, 1:3] = 0

plt.imshow(rgb)
plt.show()

cv2.imwrite('/home/andrea/Downloads/Final RGB HR Imagery/UH_NAD83_271460_3289689_centerline.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
#matplotlib.image.imsave('/home/andrea/centerline.png', rgb)
