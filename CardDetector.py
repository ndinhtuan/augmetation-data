import cv2
import numpy as np 
import copy
from help import four_point_transform

class CardDetector(object):
    'CardDetector is used for detecting card from raw image'

    def __init__(self, config=[1, 5, 0, -1, 3, \
                        cv2.RETR_LIST, cv2.cv.CV_CHAIN_APPROX_TC89_L1]):

        self.__channelImg = config[0]
        self.__sizeGaussion = config[1] 
        self.__sigmaXGaussion = config[2]
        self.__depthLaplacian = config[3]
        self.__sizeLaplacian = config[4]
        self.__modeFindContour = config[5]
        self.__methodFindContour = config[6]

    # function detectCard get id card from id card 
    # @img is raw image 
    # return card (after apply perspective transfomation) and return 
    #       image after draw contour.
    def detectCard(self, img):

        imgProcessed = self.preprocessing(img)
        imgContainCountour = copy.deepcopy(img)

        contours, _ = cv2.findContours(imgProcessed, self.__modeFindContour, 
                                        self.__methodFindContour)

        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
        screenCnt = None
        # loop over the contours
        for c in contours:
    	    # approximate the contour
    	    peri = cv2.arcLength(c, True)
    	    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    	    # if our approximated contour has four points, then we
    	    # can assume that we have found our screen
    	    if len(approx) == 4:
    	        screenCnt = approx

        if screenCnt     is not None : 
            pts = screenCnt.reshape(4, 2)
        else :
            return None, None
        warpedImg = four_point_transform(img, pts)
        cv2.drawContours(imgContainCountour, [screenCnt], -1, (0, 255, 0), 2)
        return warpedImg, imgContainCountour

    def preprocessing(self, img):

        dst = copy.deepcopy(img);
        dst = dst[:, :, self.__channelImg]
        dst = cv2.equalizeHist(dst)
        dst = cv2.GaussianBlur(dst, (self.__sizeGaussion, self.__sizeGaussion), self.__sigmaXGaussion)
        #dst = cv2.bilateralFilter(dst, 11, 17, 17)
        dst = cv2.Laplacian(dst, self.__depthLaplacian, self.__sizeLaplacian)

        return dst

    def testConfig(self, config, data):
        
        model = self.createTempDetector(config)
        results = []

        for i in range(len(data)):
            warpedImg, imgContainCountour = model.detectCard(data[i])
            results.append(warpedImg)

        return results

    def createTempDetector(self, config):

        model = CardDetector(config)
        return model
    
    def changeConfig(self, config):

        self.__channelImg = config[0]
        self.__sizeGaussion = config[1] 
        self.__sigmaXGaussion = config[2]
        self.__depthLaplacian = config[3]
        self.__sizeLaplacian = config[4]
        self.__modeFindContour = config[5]
        self.__methodFindContour = config[6]

import sys

if __name__ == "__main__" :
    
    linkImg = sys.argv[1]
    img = cv2.imread(linkImg)
    detector = CardDetector()
    data = [] 
    data.append(img) 
    config = [1, 11, 2, -1, 4, \
                        cv2.RETR_LIST, cv2.cv.CV_CHAIN_APPROX_TC89_L1]
    results = detector.testConfig(config, data) 

    if results[0] is not None : cv2.imshow("tuan", results[0])
    else :
        print "Fail detect."
    cv2.waitKey(0)