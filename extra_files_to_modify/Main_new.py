import sys,os,time,csv,getopt,cv2,argparse
import numpy as np
import shapefile
from datetime import datetime
from osgeo import gdal
from ObjectWrapper import *
from Visualize import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', dest='graph', type=str,
                        default='graph', help='MVNC graphs.')
    parser.add_argument('--image', dest='image', type=str,
                        default='./test_images/images/Output_289.jpg', help='An image path.')
    parser.add_argument('--video', dest='video', type=str,
                        default='./videos/car.avi', help='A video path.')
    parser.add_argument('--shape', dest='shape', type=str,
                        default='./test_images/shape_files/crop_289.shp/structures_poly_35_conv', help='A shape file path.')
    parser.add_argument('--tif', dest='tif', type=str,
                        default='./test_images/tifs/crop_289.tif', help='A shape file path.')
    args = parser.parse_args()

    network_blob=args.graph
    imagefile = args.image
    videofile = args.video
    shapefilepack= args.shape
    tiffile = args.tif

    detector = ObjectWrapper(network_blob)
    stickNum = ObjectWrapper.devNum

    if sys.argv[1] == '--image':
        # image preprocess
        img = cv2.imread(imagefile)
        # open tif file
        dataset = gdal.Open(tiffile)
        xOrigin, pixelWidth, xskew, yOrigin, yskew, pixelHeight = dataset.GetGeoTransform()

        # Need to convert lat,long coordinates to pixels
        def coord2pix(geox, geoy):
            px = (geox - xOrigin) / pixelWidth
            py = (geoy - yOrigin) / pixelHeight
            return px, py
        
        # open shape file, and get list of shapes
        print("SHAPE FILE PATH: " , shapefilepack)
        sf = shapefile.Reader(shapefilepack)
        box_list = []
        for shape in sf.shapes() : 
            bbox = list(shape.bbox)
            px1, py1 = coord2pix(bbox[0], bbox[1])
            px2, py2 = coord2pix(bbox[2], bbox[3] )
            box = (int(px1), int(py2), int(px2), int(py1))
            box_list.append(box)
            # draw a green rectangle to visualize the bounding rect
            #cv2.rectangle(img, (int(px1), int(py2)), (int(px2), int(py1)), (0, 255, 0), 2)

  
        print("I have ", len(box_list), " many shapes")
        
        def bb_intersection_over_union(boxA, boxB):
            #print("box A " , boxA)
            #print("box B " , boxB)
            #boxA = list(boxA)
            #boxB = list(boxB)

            left = boxB.left
            top = boxB.top
            right = boxB.right
            bottom = boxB.bottom
            #print("coords:" , left, top, right, bottom)

            # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(boxA[0], left)
            yA = max(boxA[1], top)
            xB = min(boxA[2], right)
            yB = min(boxA[3], bottom)

            if xB < xA or yB < yA :
                return 0

	    # compute the area of intersection rectangle
            interArea = (xB - xA + 1) * (yB - yA + 1)
 
	    # compute the area of both the prediction and ground-truth
	    # rectangles
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (right - left + 1) * (bottom - top + 1)
 
            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            iou = interArea / float(boxAArea + boxBArea - interArea)
            # return the intersection over union value
            return iou

        start = datetime.now()

        results = detector.Detect(img)

        end = datetime.now()
        elapsedTime = end-start

        print ('total time is " milliseconds', elapsedTime.total_seconds()*1000)
        iou_list = []
        # Calculate IOU. will need to make some assumptions here
        for box in box_list:
            max_iou = 0
            for pred in results:
                iou = bb_intersection_over_union(box, pred)
                if iou > max_iou and iou > 0.4:
                   iou = round(iou,2)
                   max_iou = iou
            print("IOU: ", max_iou)
            iou_list.append(max_iou)

        for box, ious in zip(box_list,iou_list):
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            #print(box, " ", ious)
            #cv2.rectangle(img, (left,top), (right,bottom), clr, thickness=3)
            #cv2.rectangle(img, (left,top-20),(right,top),(255,255,255),-1)
            cv2.putText(img,str(ious),(box[0]+5,box[1]-7),cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255), 2)
        
        imdraw = Visualize(img, results)
        cv2.imshow('Demo',imdraw)
        cv2.imwrite('test.jpg',imdraw)
        cv2.waitKey(10000)
    elif sys.argv[1] == '--video':
        # video preprocess
        cap = cv2.VideoCapture(videofile)
        fps = 0.0
        while cap.isOpened():
            start = time.time()
            imArr = {}
            results = {}
            for i in range(stickNum):
                ret, img = cap.read()
                if i not in imArr:
                    imArr[i] = img
            if ret == True:
                tmp = detector.Parallel(imArr)
                for i in range(stickNum):
                    if i not in results:
                        results[i] = tmp[i]
                    imdraw = Visualize(imArr[i], results[i])
                    fpsImg = cv2.putText(imdraw, "%.2ffps" % fps, (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    cv2.imshow('Demo', fpsImg)
                end = time.time()
                seconds = end - start
                fps = stickNum / seconds
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
