import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt
import cv2
import matplotlib
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Prepare the houston dataset for roadnet")
parser.add_argument('--image', type=str, help="path to a tif image to be converted to png")
args = parser.parse_args()

img_path = Path(args.image)

ds = gdal.Open(img_path.as_posix())
myarray = np.array(ds.GetRasterBand(1).ReadAsArray(), dtype=np.uint8)
myarray[myarray==1] = 255
plt.imshow(myarray, cmap='gray')
plt.show()

rgb = np.full(myarray.shape + (3,), 255, dtype=np.uint8)
rgb[myarray==255, 1:3] = 0

plt.imshow(rgb)
plt.show()

out_path = img_path.as_posix().replace('.tif', '.png')
print(out_path)
cv2.imwrite(out_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
