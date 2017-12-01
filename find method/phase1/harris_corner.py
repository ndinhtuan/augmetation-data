import cv2 
import sys
import numpy as np

if __name__ == "__main__":
    path_img = sys.argv[1]
    img = cv2.imread(path_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 3, 3, 0.04)
    dst = cv2.dilate(dst, None)

    cv2.imshow("dst", dst)
    cv2.waitKey(0)

    #img[dst>0.01*dst.max()]=[0,0,255] detect corner (R large)
    img[dst < 0] = [0, 0, 255]
    if len(sys.argv) >= 3 : 
        cv2.imwrite(sys.argv[2], img)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()