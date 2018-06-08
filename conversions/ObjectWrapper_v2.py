from libpydetector import YoloDetector
import os, io, numpy, time
import numpy as np
from mvnc import mvncapi as mvnc
from skimage.transform import resize

class BBox(object):
    def __init__(self, bbox, xscale, yscale, offx, offy):
        self.left = int(bbox.left / xscale)-offx
        self.top = int(bbox.top / yscale)-offy
        self.right = int(bbox.right / xscale)-offx
        self.bottom = int(bbox.bottom / yscale)-offy
        self.confidence = bbox.confidence
        self.objType = bbox.objType
        self.name = bbox.name

class ObjectWrapper():
    # open device
    mvnc.global_set_option(mvnc.GlobalOption.RW_LOG_LEVEL, 2)
    devices = mvnc.enumerate_devices()
    devNum = len(devices) # will have more than one probably
    print("Number of Movidius Sticks detected : " , devNum)
    if len(devices) == 0:
        print('No MVNC devices found')
        quit()
    devHandle = [] # used as device list - store devices
    graphHandle = [] # used as graph list - store graphs
    inputHandle = []
    outputHandle = []
    def __init__(self, graphfile):
        select = 1
        self.detector = YoloDetector(select)

        for i in range(ObjectWrapper.devNum): # will loop for each device detected
            ObjectWrapper.devHandle.append(mvnc.Device(ObjectWrapper.devices[i])) # pass in list of devices, append that device to device list.
            ObjectWrapper.devHandle[i].open() # open that device.
            #opt = ObjectWrapper.devHandle[i].GetDeviceOption(mvnc.DeviceOption.OPTIMISATION_LIST)
            # load blob
            with open(graphfile, mode='rb') as f:
                blob = f.read()
            graph = mvnc.Graph('graph1') # creates a graph instance

            # Allocate the graph and store to array
            #ObjectWrapper.graphHandle.append(graph.allocate(ObjectWrapper.devHandle[i], blob))
            input_fifo, output_fifo = graph.allocate_with_fifos(ObjectWrapper.devHandle[i], blob)

            ObjectWrapper.graphHandle.append(graph)
            ObjectWrapper.inputHandle.append(input_fifo)
            ObjectWrapper.outputHandle.append(output_fifo)

            self.dim = (416,416)
            self.blockwd = 12
            self.wh = self.blockwd*self.blockwd
            self.targetBlockwd = 13
            self.classes = 1
            self.threshold = 0.2
            self.nms = 0.4


    #def __del__(self):

    def PrepareImage(self, img, dim):
        imgw = img.shape[1]
        imgh = img.shape[0]
        imgb = np.empty((dim[0], dim[1], 3))
        imgb.fill(0.5)

        if imgh/imgw > dim[1]/dim[0]:
            neww = int(imgw * dim[1] / imgh)
            newh = dim[1]
        else:
            newh = int(imgh * dim[0] / imgw)
            neww = dim[0]
        offx = int((dim[0] - neww)/2)
        offy = int((dim[1] - newh)/2)

        imgb[offy:offy+newh,offx:offx+neww,:] = resize(img.copy()/255.0,(newh,neww),1)
        im = imgb[:,:,(2,1,0)]
        return im, int(offx*imgw/neww), int(offy*imgh/newh), neww/dim[0], newh/dim[1]

    def Reshape(self, out, dim):
        shape = out.shape
        out = np.transpose(out.reshape(self.wh, int(shape[0]/self.wh)))  
        out = out.reshape(shape)
        return out

    def Detect(self, img):
        print("DOING SINGLE DETECT")
        imgw = img.shape[1]
        imgh = img.shape[0]

        im,offx,offy,xscale,yscale = self.PrepareImage(img, self.dim)
        
        ####Edit
        ObjectWrapper.graphHandle[0].queue_inference_with_fifo_elem(ObjectWrapper.inputHandle[0], ObjectWrapper.outputHandle[0], im.astype(np.float32), 'user object')
        out, userobj = ObjectWrapper.outputHandle[0].read_elem() # Get result from output queue
        ####

        out = self.Reshape(out, self.dim)

        internalresults = self.detector.Detect(out.astype(np.float32), int(out.shape[0]/self.wh), self.blockwd, self.blockwd, self.classes, imgw, imgh, self.threshold, self.nms, self.targetBlockwd)
        pyresults = [BBox(x,xscale,yscale, offx, offy) for x in internalresults]
        print(pyresults)
        return pyresults

    def Parallel(self, img):
        print("DOING PARALLEL")
        pyresults = {}
        for i in range(ObjectWrapper.devNum):
            im, offx, offy, w, h = self.PrepareImage(img[i], self.dim)
            # Edit
            ObjectWrapper.graphHandle[i].queue_inference_with_fifo_elem(ObjectWrapper.inputHandle[i], ObjectWrapper.outputHandle[0], im.astype(np.float32), 'user object')
            #ObjectWrapper.graphHandle[i].LoadTensor(im.astype(np.float16), 'user object')

        for i in range(ObjectWrapper.devNum):
            # Edit
            out, userobj = ObjectWrapper.outputHandle[i].read_elem() # Get result from output queue
            #out, userobj = ObjectWrapper.graphHandle[i].GetResult()

            out = self.Reshape(out, self.dim)
            imgw = img[i].shape[1]
            imgh = img[i].shape[0]
            internalresults = self.detector.Detect(out.astype(np.float32), int(out.shape[0]/self.wh), self.blockwd, self.blockwd, self.classes, imgw, imgh, self.threshold, self.nms, self.targetBlockwd)
            res = [BBox(x, w, h, offx, offy) for x in internalresults]
            if i not in pyresults:
                pyresults[i] = res
        return pyresults
