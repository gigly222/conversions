import glob, os

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Directory where the data will reside, relative to 'darknet.exe'
path_data = 'data/obj/'

# Percentage of images to be used for the test set
percentage_test = 10;

# Create and/or truncate train.txt and test.txt
file_train = open('train.txt', 'w')
file_test = open('test.txt', 'w')

# Populate train.txt and test.txt
counter = 1
index_test = round(100 / percentage_test)
for pathAndFilename in glob.iglob(os.path.join(current_dir, "*.jpg")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))

    if counter == index_test:
        counter = 1
        file_test.write(path_data + "images/" + title + '.jpg' + "\n")
    else:
        file_train.write(path_data + "images/" + title + '.jpg' + "\n")
        counter = counter + 1

#  ./darknet detector train cfg/house.data cfg/yolo-obj.cfg darknet19_448.conv.23
# ./darknet detector test cfg/obj.data cfg/yolo-obj.cfg backup/yolo-obj_300.weights data/obj/images/Output_33.jpg
