import sys,os
from Utility_Methods import convert_cords, create_dirs, chunk_images
from Yolo_Formatter import convert_to_yolo
# need to convert jp2 to tif first at the moment.
# must provide original shape file, path to save convert coordinate shape file, this folder directory path, jp2 path and matching tif path.

if __name__ == "__main__":

    # check for enough arguments. Need shape file path, and an empty file name to store shape file in like /PycharmProjects/structures_poly_35_conv.shp
    if len(sys.argv) != 6:
        print("Need input shape file path and the output path you want filed stored as...")
        sys.exit(1)

    # Get shapefile path and output path
    input_path=sys.argv[1]
    conv_shape_path=sys.argv[2]
    dir_path = sys.argv[3] # dir path to current directory where code exists. This is were mini folders created
    jp2_path = sys.argv[4]
    tiff_path = sys.argv[5]

    # Convert coordinate system of shapefile
    if os.path.exists(input_path) and os.path.exists(conv_shape_path):
        # Convert shape file to new coordinate system.
        print("Converting shape files to new coordinate system...")
        convert_cords(input_path, conv_shape_path)
    else:
        print("Either the input path or the output path does not exist!")
        sys.exit(1)

    # Make directories to Store chunked peices needed for yolo
    new_jp2_path, new_tif_path, new_shapefile_path, new_bb_path = create_dirs(dir_path)

    # convert jp2 to tiff image

    # Chunk jp2 image, convert chunks to tif image, crop shape file to tif size, save jp2,tif and crop shape files each to different folder.
    print("chunking images")
    chunk_images(jp2_path, tiff_path, conv_shape_path, new_jp2_path, new_tif_path, new_shapefile_path, new_bb_path)

    # Run yolo conversion script on bounding boxes.
    convert_to_yolo()


# EX: /home/datascientist/PycharmProjects/Scout/Boston_Shape_Files/structures_poly_35.shp /home/datascientist/PycharmProjects/Scout/Boston_Shape_Files/structures_poly_35_conv.shp /home/datascientist/PycharmProjects/Scout/Create_House_Images/ /home/datascientist/PycharmProjects/Scout/Boston_Images/19TCG315890/19TCG315890.jp2 /home/datascientist/PycharmProjects/Scout/Boston_Images/19TCG315890/19TCG315890.tif