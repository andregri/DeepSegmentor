import processing
import tempfile

data_dir = "/home/andrea/Downloads/Final RGB HR Imagery/2/"

# Load raster layer
path_to_tif = data_dir + "UH_NAD83_271460_3290290.tif"
rlayer = QgsRasterLayer(path_to_tif, "UH_NAD83_271460_3290290")
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
path_to_centerline_layer = data_dir + "UH_NAD83_271460_3290290_centerline.kml"
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