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
from pathlib import Path

parser = argparse.ArgumentParser(description="Prepare the houston dataset for roadnet")
parser.add_argument('--data_dir', type=str, default="/home/andrea/Downloads/Final RGB HR Imagery/5", help="dir containing the image")

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

data_dir = Path(args.data_dir)
number = args.data_dir.split('/')[-1]
tif_file = [img for img in data_dir.rglob('UH_NAD*.tif')][0]
osm_file   = [o for o in data_dir.rglob('centerline.geojson')][0]
print(data_dir)
print(tif_file)
print(osm_file)

# Load raster layer
rlayer = QgsRasterLayer(tif_file.as_posix(), tif_file.name.replace('.tif', ''))
if not rlayer.isValid():
    print("Layer failed to load!")
else:
    QgsProject.instance().addMapLayer(rlayer)

# Resample the raster layer and save to a file
path_to_resampled = (data_dir / "Houston-{}.tif".format(number)).as_posix()
print(path_to_resampled)
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
vlayer = QgsVectorLayer(osm_file.as_posix(), "centerline", "ogr")
if not vlayer.isValid():
    print("Layer failed to load: vlayer")
else:
    QgsProject.instance().addMapLayer(vlayer)
    
# Reproject the vector layer
#temp_dir = tempfile.TemporaryDirectory()
parameters = {
    "INPUT": vlayer,
    "TARGET_CRS": "EPSG:26915",
    "OUTPUT": 'memory:Reprojected'
}
reproj_layer = processing.run("native:reprojectlayer", parameters)['OUTPUT']
if not reproj_layer.isValid():
    print("Layer failed to load: reproj layer")
else:
    QgsProject.instance().addMapLayer(reproj_layer)
    
# Rasterize the vector layer
path_to_rasterization = (data_dir / "centerline.tif").as_posix()
parameters = {
    "INPUT": reproj_layer,
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
    print("Layer failed to load: rasterized layer")
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