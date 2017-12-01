import cv2 
import sys
import numpy as np

if __name__ == "__main__":
    path_img = sys.argv[1]
    img = cv2.imread(path_img)
    t = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fast = cv2.FastFeatureDetector()
    kp = fast.detect(t, None)
    t = cv2.drawKeypoints(t, kp, color=(255, 0, 0))
    if len(sys.argv) >= 3 : 
        cv2.imwrite(sys.argv[2], img)
    cv2.imshow("img", t)
    cv2.waitKey(0)
    cv2.destroyAllWindows()