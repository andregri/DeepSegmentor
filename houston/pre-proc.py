import sys
import tempfile

from qgis.core import (
     QgsApplication, 
     QgsProcessingFeedback, 
     QgsVectorLayer,
     QgsRasterLayer,
     QgsProject
)

from qgis.analysis import QgsNativeAlgorithms

import argparse

parser = argparse.ArgumentParser(description="Prepare the houston dataset for roadnet")
parser.add_argument('--data_dir', type=str, default="/home/andrea/Downloads/Final RGB HR Imagery/3/", help="dir containing the image")
parser.add_argument('--image', type=str, default="UH_NAD83_271460_3290290.tif", help="filename of the tif image")
parser.add_argument('--osm', type=str, default="UH_NAD83_271460_3290290_centerline.kml", help="filename of the kml file downloaded from overpass")

args = parser.parse_args()

# See https://gis.stackexchange.com/a/155852/4972 for details about the prefix 
QgsApplication.setPrefixPath('/usr', True)
qgs = QgsApplication([], False)
qgs.initQgis()

# Append the path where processing plugin can be found
sys.path.append('/usr/share/qgis/python/plugins')

import processing
from processing.core.Processing import Processing
Processing.initialize()
QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())

data_dir = args.data_dir + '/'
print(data_dir)
image = args.image
print(image)
osm = args.osm
print(osm)

# Load raster layer
path_to_tif = data_dir + image
rlayer = QgsRasterLayer(path_to_tif, image.replace('.tif', ''))
if not rlayer.isValid():
    print("Layer failed to load!")
else:
    QgsProject.instance().addMapLayer(rlayer)

# Resample the raster layer and save to a file
path_to_resampled = data_dir + "resampled.tif"
parameters = {
    "INPUT": rlayer,
    "RESAMPLING": 0,
    "TARGET_RESOLUTION": 0.21,
    "DATA_TYPE": 0,
    "TARGET_EXTENT": rlayer.extent(),
    "OUTPUT": path_to_resampled
}
processing.run("gdal:warpreproject", parameters)

resampled_layer = QgsRasterLayer(path_to_resampled, "resampled_tif")
if not resampled_layer.isValid():
    print("Layer failed to load!")
else:
    QgsProject.instance().addMapLayer(resampled_layer)

# Load vector layer
path_to_centerline_layer = data_dir + osm
vlayer = QgsVectorLayer(path_to_centerline_layer, "centerline", "ogr")
if not vlayer.isValid():
    print("Layer failed to load!")
else:
    QgsProject.instance().addMapLayer(vlayer)
    
# Reproject the vector layer
temp_dir = tempfile.TemporaryDirectory()
path_to_reproj = temp_dir.name + "/reproj.shp"
parameters = {
    "INPUT": vlayer,
    "TARGET_CRS": "EPSG:26915",
    "OUTPUT": path_to_reproj
}
processing.run("qgis:reprojectlayer", parameters)
reproj_layer = QgsVectorLayer(path_to_reproj, "reproj", "ogr")
if not reproj_layer.isValid():
    print("Layer failed to load!")
else:
    QgsProject.instance().addMapLayer(reproj_layer)
    
# Rasterize the vector layer
path_to_rasterization = data_dir + "centerline.tif"
parameters = {
    "INPUT": path_to_reproj,
    "BURN": 1,
    "UNITS": 1,
    "WIDTH": 0.21,
    "HEIGHT": 0.21,
    "EXTENT": resampled_layer.extent(),
    "DATA_TYPE": 0,
    "OUTPUT": path_to_rasterization
}
processing.run("gdal:rasterize", parameters)
rasterized_layer = QgsRasterLayer(path_to_rasterization, "rasterization")
if not rasterized_layer.isValid():
    print("Layer failed to load!")
else:
    QgsProject.instance().addMapLayer(rasterized_layer)

# Convert tif to png
import numpy as np
from osgeo import gdal
import cv2

ds = gdal.Open(path_to_rasterization)
myarray = np.array(ds.GetRasterBand(1).ReadAsArray(), dtype=np.uint8)
myarray[myarray==1] = 255

rgb = np.full(myarray.shape + (3,), 255, dtype=np.uint8)
rgb[myarray==255, 1:3] = 0

path_to_png = path_to_rasterization.replace('.tif', '.png')
cv2.imwrite(path_to_png, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))