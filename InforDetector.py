import cv2
import numpy as np 
import copy
from WordDetector import WordDetector
from utils_shape import classify_box, get_area_intersection, remove_box_inside_other_box, get_area_of_box
import pickle
from extract_information import connect_bounds, connect

class InforDetector(object):
    'InforDetector is used for dectecting information in identity card'
    
    def __init__(self):
        return

    def detect_infors(self, warpedImg):

        with open("ratios.txt", "rb") as fp:
            ratios = pickle.load(fp)

        infors = [] # consisting all infors field in Identity Card
        detectWord = WordDetector()
        wordImages, boxes = detectWord.detectWord(warpedImg)
        self.eliminate_not_box(boxes)
        self.norm_boxes(boxes)

        classified = []
        h, w, _ = warpedImg.shape
        classify_box(boxes, ratios, h, w, classified)
        for classi, i in zip(classified, range(len(classified))):
        
            classi = connect_bounds(classi, warpedImg)
            self.eliminate_small_area_box(classi)
            remove_box_inside_other_box(classi)
            self.join_infor(classi)
            for box in classi:
                
                infors.append(warpedImg[box[1]:box[1]+box[3], box[0]:box[0]+box[2]])
        return infors

    def eliminate_not_box(self, boxes):
        not_box = []

        for i in range(len(boxes)):
            
            if boxes[i][3] * 1.0 / boxes[i][2] > 2: 
                not_box.append(i)

        for i in range(len(not_box)):
            del boxes[not_box[len(not_box) - 1- i]]

    def norm_boxes(self, boxes):
        threshold_h = 70
        barrier = 3

        for i in range(len(boxes)):
            if boxes[i][3] > threshold_h:
                old_h = boxes[i][3]
                boxes[i] = [boxes[i][0], boxes[i][1], boxes[i][2], old_h /2-barrier] #box1
                box2 = [boxes[i][0], boxes[i][1] + old_h / 2 + barrier, boxes[i][2], old_h/2]
                boxes.insert(i, box2)
                i = i + 1
    
    def eliminate_small_area_box(self, boxes):

        threshold_area = 500

        i = 0
        while i < len(boxes):
            if (get_area_of_box(boxes[i]) < threshold_area): del boxes[i]
            i += 1

    def join_infor(self, boxes):

        alpha = 0.3

        i = 0
        while i < len(boxes):
            j = 0
            while j < len(boxes):
                if (i != j): 
                    
                    area_intersection = get_area_intersection(boxes[i], boxes[j])
                    area_min = min(get_area_of_box(boxes[i]), get_area_of_box(boxes[j]))

                    if area_intersection >= alpha * area_min:
                        boxes[i] = connect(boxes[i], boxes[j])
                        del boxes[j]
                    else :
                        j += 1
                    continue
                j += 1
            i += 1

import sys
from CardDetector import CardDetector

if __name__ == "__main__":

    linkImg = sys.argv[1]
    img = cv2.imread(linkImg)
    detector = CardDetector(None)
    result, _ = detector.detectCard(img)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    inforDetector = InforDetector()
    infors = inforDetector.detect_infors(result)

    for infor in infors:
        cv2.imshow("infor", infor)
        cv2.waitKey(0)