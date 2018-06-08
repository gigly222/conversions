# Import Modules
from osgeo import ogr, osr
import shapefile
import cv2
from gdalconst import *
import numpy as np
import sys, os
from osgeo import gdal



# Converts shape files into specified coordinate system. Must provide original coordinate sys and new coordinate sys.
def crop_file(image_file, shape_file, cropped_shape_file):

    dataset = gdal.Open(image_file)
    # Get lrx, lry which is the lower right corner. These are used to crop the shape file to be the same size as the tif image of interest.
    xOrigin, pixelWidth, xskew, yOrigin, yskew, pixelHeight = dataset.GetGeoTransform()
    lrx = xOrigin + (dataset.RasterXSize * pixelWidth)
    lry = yOrigin + (dataset.RasterYSize * pixelHeight)

    # Tif Image Data
    os.system("ogr2ogr -f \'ESRI Shapefile\' -clipdst " + str(xOrigin) + " " + str(yOrigin) + " " + str(lrx) + " " + str(lry) + " " + cropped_shape_file + " " + shape_file)



# Chunk Images into smaller pieces.
def chunk_images(image_path, tif_path, shape_path, new_jp2_path, new_tif_path, new_shapefile_path, new_bb_path):

    print("start chunking")

    dataset = gdal.Open(tif_path)

   # image_proj = dataset.GetProjection()
    band1 = dataset.GetRasterBand(1)  # split into RGB bands. Used to create color image.
    band2 = dataset.GetRasterBand(2)
    band3 = dataset.GetRasterBand(3)

    # Create color bands and re-create color images
    red = band1.ReadAsArray()
    green = band2.ReadAsArray()
    blue = band3.ReadAsArray()
    #image = np.array(dataset.ReadAsArray(), dtype=float)
    rgbArray = np.dstack((red, green, blue)) # combine channels to get color image
    image = rgbArray # Now we have the color image

    # used fo cropping and naming images
    (winW, winH) = (416, 416)  # window sizes
    count = 289  # used in naming cropped files

    # Sliding window of both jpg and tif raster image.
    def sliding_window(image, stepSize, windowSize):
        # slide a window across the image
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                # yield the current window
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

    # loop over as sliding window over color image. Each cropped image we check # of shape file white pixels.
    for (x, y, window) in sliding_window(image, stepSize=416, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        # Copy and crop images and tiff file
        #clone = rasterArray.copy()
        cloneImage = image.copy()

        cropped_tif = new_tif_path + '/crop_' + str(count) + ".tif"
        if not os.path.exists(cropped_tif):
            os.mknod(cropped_tif)

        # crops tif file and saves it to folder
        gdal.Translate(cropped_tif, dataset, srcWin=[x, y,  winW ,  winH])

        new_shapefile_dir = new_shapefile_path + '/crop_' + str(count) + ".shp"
        if not os.path.exists(new_shapefile_dir):
            os.mkdir(new_shapefile_dir)

        # Crops shapefile and saves it to folder
        crop_file(cropped_tif, shape_path, new_shapefile_dir)

        stringLine = new_jp2_path + "/Output_" + str(count) + '.jpg'
        crop_img = cloneImage[y: y + winH,
                   x: x + winW]  # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]

        get_bounding_boxes(cropped_tif, new_shapefile_dir, count, new_bb_path)

        cv2.imwrite( stringLine, crop_img )
        count += 1
        #cv2.imshow('image', crop_img)
        #cv2.waitKey(1)


# Get Bounding Boxes in pixel form from shape files
def get_bounding_boxes(image_path, shape_path, count, new_bb_path):
    # print("PATH ", image_path)
    # print("Shape PATH ", shape_path)
    # Read in image for display purposes to check you are actually making the correct conversion for bbox
    img = cv2.imread(image_path)

    # Need to convert lat,long coordinates to pixels
    def coord2pix(geox, geoy):
        px = (geox - xOrigin) / pixelWidth
        py = (geoy - yOrigin) / pixelHeight
        return px, py

    # Reads in tif image to get information about pixel size, origin...will need these for converting lat and long to pixels
    dataset = gdal.Open(image_path)

    # Reads in cropped shape file
    sf = shapefile.Reader(shape_path + "/structures_poly_35_conv.shp")

    # Get lrx, lry which is the lower right corner. These are used to crop the shape file to be the same size as the tif image of interest.
    xOrigin, pixelWidth, xskew, yOrigin, yskew, pixelHeight = dataset.GetGeoTransform() # taken from tif image.

    bb_output_txt =new_bb_path + "/Output_" + str(count) + ".txt"
    with open(bb_output_txt, "w") as text_file:

        # loop through shape files and get bounding box of each. Then convert bounding box from lat/long to pixels.
        for shape in sf.shapes():
            bbox = list(shape.bbox) # lower left (x,y) coordinate and upper right corner coordinate ...[x-orgin, y-origin, lrx, lry]
            px1, py1 = coord2pix(bbox[0], bbox[1])
            px2, py2 = coord2pix(bbox[2], bbox[3] )

            print("px1 ", px1)
            print("y1 ", py1)
            print("px2 ", px2)
            print("py2 ", py2)
            # draw a green rectangle to visualize the bounding rect
            cv2.rectangle(img, (int(px1), int(py2)), (int(px2), int(py1)), (0, 255, 0), 2)
            box = (px1, py2, px2, py1) # should be upper left and lower right
            text_file.write("0" + " " +" ".join([str(a) for a in box]) + '\n')
           # print(bb)

        # Check you bounding boxes align with houses
        cv2.imshow('image',img)
        k = cv2.waitKey(1)


# Create directories to store cropped jp2, tifs and shapefiles
def create_dirs(dir_path):
    new_jp2_path = dir_path + "crop_jp2_files"
    new_tif_path = dir_path + "crop_tif_files"
    new_shapefile_path = dir_path + "crop_shape_files"
    new_bb_path = dir_path + "crop_bb_files"

    if not os.path.exists(new_jp2_path):
        os.makedirs(new_jp2_path)

    if not os.path.exists(new_tif_path):
        os.makedirs(new_tif_path)

    if not os.path.exists(new_shapefile_path):
        os.makedirs(new_shapefile_path)

    if not os.path.exists(new_bb_path):
        os.makedirs(new_bb_path)

    return new_jp2_path, new_tif_path, new_shapefile_path, new_bb_path

# Converts shape files into specified coordinate system.
# Must provide original coordinate sys and new coordinate sys.
def convert_cords(infile, outfile):
    # get path and filename seperately
    (out_file_path, out_file_name) = os.path.split(outfile)

    # get file name without extension
    (out_file_short_name, extension) = os.path.splitext(out_file_name)  # get file name without extension

    # Spatial Reference of the input file. (Need to l=/usr/include/gdalook at the image to see what coordinate system it is in)
    # Access the Spatial Reference and assign the input projection
    in_SpatialRef = osr.SpatialReference()
    in_SpatialRef.ImportFromEPSG(26986)

    # Spatial Reference of the output file
    # Access the Spatial Reference and assign the output projection (Need to look up the output coordinate system you want)
    out_SpatialRef = osr.SpatialReference()
    out_SpatialRef.ImportFromEPSG(26919)

    # create Coordinate Transformation
    coordTransform = osr.CoordinateTransformation(in_SpatialRef, out_SpatialRef)

    # Open the input shapefile and get the layer
    driver = ogr.GetDriverByName('ESRI Shapefile')

    in_dataset = driver.Open(infile, 0)

    if in_dataset is None:
        print('Could not open file')
        sys.exit(1)

    inlayer = in_dataset.GetLayer()

    # Create the output shapefile but check first if file exists
    if os.path.exists(outfile):
        driver.DeleteDataSource(outfile)

    out_dataset = driver.CreateDataSource(outfile)

    if outfile is None:
        print('Could not create file')
        sys.exit(1)
    out_layer = out_dataset.CreateLayer(out_file_short_name, geom_type=ogr.wkbPolygon)

    # Get the FieldDefn for attributes and add to output shapefile
    feature = inlayer.GetFeature(0)
    feild_array = []
    feild_name_array = []
    for i in range(feature.GetFieldCount()):
        name_of_feature = feature.GetDefnRef().GetFieldDefn(i).GetName()
        # print(name_of_feature)
        feild_name_array.append(name_of_feature)  # Gets list of all feature names
        feild_array.append(feature.GetDefnRef().GetFieldDefn(i))  # Now we have all feature names for all attributes

    for i in range(len(feild_array)):
        # print(type(featureArray[i]))
        out_layer.CreateField(feild_array[i])

    # get the FeatureDefn for the output shapefile
    feature_defn = out_layer.GetLayerDefn()

    # Loop through input features and write to output file
    in_feature = inlayer.GetNextFeature()
    while in_feature:
        # get the input geometry
        geometry = in_feature.GetGeometryRef()

        # re-project the geometry, each one has to be projected separately
        geometry.Transform(coordTransform)

        # create a new output feature
        out_feature = ogr.Feature(feature_defn)

        # set the geometry and attribute
        out_feature.SetGeometry(geometry)

        # Get features
        for i in range(len(feild_name_array)):
            out_feature.SetField(feild_name_array[i], in_feature.GetField(feild_name_array[i]))

        out_layer.CreateFeature(out_feature)

        # destroy the features and get the next input features
        out_feature.Destroy
        in_feature.Destroy
        in_feature = inlayer.GetNextFeature()

    # close the shapefiles
    in_dataset.Destroy()
    out_dataset.Destroy()

    # create the prj projection file
    out_SpatialRef.MorphToESRI()
    file = open(out_file_path + '/' + out_file_short_name + '.prj', 'w')

    print(out_file_path + '/' + out_file_short_name + '.prj')
    file.write(out_SpatialRef.ExportToWkt())
    file.close()
