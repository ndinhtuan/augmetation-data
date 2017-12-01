import cv2 
import sys
import numpy as np

if __name__ == "__main__":
    path_img = sys.argv[1]
    img = cv2.imread(path_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIF()
    kp = sift.detect(gray, None)
    img = cv2.drawKeypoints(gray, kp)
    if len(sys.argv) >= 3 : 
        cv2.imwrite(sys.argv[2], img)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()