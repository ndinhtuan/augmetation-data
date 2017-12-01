import cv2
import numpy as np 
import copy
from help import four_point_transform
import json

class CardDetector(object):
    'CardDetector is used for detecting card from raw image'

    def __init__(self, config=[[60, 20, 20], [100, 255, 255], (10, 10), \
                        cv2.RETR_LIST, cv2.cv.CV_CHAIN_APPROX_TC89_L1]):

        if (config is not None):
            self.__lowRange = config[0]
            self.__highRange = config[1] 
            self.__maskDilating = config[2]
            self.__modeFindContour = config[3]
            self.__methodFindContour = config[4]
        else :
            with open('configDetector.json', 'r') as f:
                config = json.load(f)
                self.__lowRange = config["lowRange"]
                self.__highRange = config["highRange"] 
                self.__maskDilating = config["maskDilating"]
                self.__modeFindContour = config["modeContour"]
                self.__methodFindContour = config["methodContour"]

    # function detectCard get id card from id card 
    # @img is raw image 
    # return card (after apply perspective transfomation) and return 
    #       image after draw contour.
    def detectCard(self, img):

        img1 = copy.deepcopy(img)
        img_processed = self.pre_processing(img1)
        contours, hi = cv2.findContours(img_processed, cv2.RETR_EXTERNAL, cv2.cv.CV_CHAIN_APPROX_TC89_KCOS)

        contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
        copy_img = copy.deepcopy(img)
        rightContour = None

        peri = cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], 0.02 * peri, True)
        rect = cv2.minAreaRect(approx)
        box = cv2.cv.BoxPoints(rect)
        pts = np.int0(box)

        warpedImg = four_point_transform(img1, pts)
        cv2.drawContours(img1, [pts], -1, (0, 255, 0), 2)
        return warpedImg, img1

    def pre_processing(self, img):

        dst = copy.deepcopy(img)

        img_hsv = cv2.cvtColor(dst, cv2.cv.CV_BGR2HSV)
        mask = cv2.inRange(img_hsv, np.asarray(self.__lowRange), np.asarray(self.__highRange))
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (self.__maskDilating[0], self.__maskDilating[1]))
        mask = cv2.dilate(mask, element)
    
        return mask

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

        self.__lowRange = config[0]
        self.__highRange = config[1] 
        self.__maskDilating = config[2]
        self.__modeFindContour = config[3]
        self.__methodFindContour = config[4]
        file_config = {
        "lowRange":config[0], 
        "highRange": config[1], 
        "maskDilating":config[2],
        "modeContour": config[3], 
        "methodContour": config[4]
        }
        with open('configDetector.json', 'w') as f:
            json.dump(file_config, f)

import sys

if __name__ == "__main__" :
    
    linkImg = sys.argv[1]
    img = cv2.imread(linkImg)
    detector = CardDetector(None)
    result, _ = detector.detectCard(img)
    cv2.imshow("result", result)
    # data = [] 
    # data.append(img) 
    config = [[60, 21, 20], [100, 255, 255], (10, 10), \
                         cv2.RETR_LIST, cv2.cv.CV_CHAIN_APPROX_TC89_L1]
    detector.changeConfig(config)
    result, _ = detector.detectCard(img)
    cv2.imshow("result2", result)
    # results = detector.testConfig(config, data) 

    # if results[0] is not None : cv2.imshow("tuan", results[0])
    # else :
    #     print "Fail detect."
    cv2.waitKey(0)