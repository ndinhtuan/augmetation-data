from imgaug import augmenters as iaa
import numpy as np 
import cv2 

class AugData(object):

    def __init__(self):
        pass 

    #images is list of image (w,h,channels) of (N, w, h, c)
    # output is similar so can train 
    def augmentData(self, images):
        
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        augumenters = iaa.Sequential([
            iaa.SomeOf(n=(0, 4), children=[
                # iaa.Superpixe-ls(p_replace=(0.1, 1.0), n_segments=(10, 120)),
                iaa.Invert(0.2, 0.5),
                iaa.Add(value=(-40, 40), per_channel=0.5),
                iaa.Multiply(mul=(0.5, 1.5), per_channel=0.5),
                iaa.OneOf([
                    iaa.GaussianBlur(sigma=(0.0, 0.2)),
                    iaa.AverageBlur(k=(2, 9)),
                    iaa.MedianBlur(k=(3, 9))
                ]),
                iaa.AdditiveGaussianNoise(scale=0.05*255, per_channel=0.5),
                iaa.Dropout(p=(0, 0.2), per_channel=0.5),
                iaa.Grayscale(alpha=(0.0, 1.0)),
                iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5),
                iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))
            ], random_order=True),
            sometimes(iaa.CropAndPad(percent=(-0.05, 0.1), pad_mode=["constant", "edge"], pad_cval=(0, 128)))
        ], random_order=False)
        augumenters = augumenters.to_deterministic() #
        newDatas = augumenters.augment_images(images)
        return newDatas # -> dua vao train binh thuong

import glob
if __name__ == "__main__":
    # xem test.
    linkToImages = glob.glob("/home/tuan/Desktop/IdentityCard/Source/*.jpg")
    data = []

    for link in linkToImages[0:35]:
        img = cv2.imread(link)
        img = cv2.resize(img, dsize=None, fx=800.0/img.shape[1], fy=800.0/img.shape[0])
        if img is None : continue
        data.append(img)
    augData = AugData()
    newDatas = augData.augmentData(data)

    for img in newDatas:
        cv2.imshow("tuan", img)
        cv2.waitKey(0)
