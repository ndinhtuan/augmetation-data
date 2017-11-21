# -*- coding: utf-8 -*-
from keras import backend as K
from keras.models import Model
import numpy as np
import itertools
import cv2
from keras.models import model_from_json

alphabet =  u'abcdđefghijklmnopqrstuvwxyzABCDĐEFGHIJKLMNOPQRSTUVWXYZ àảãáạăằẳẵắặâầẩẫấậèẻẽéẹêềểễếệìỉĩíịòỏõóọôồổỗốộơờởỡớợùủũúụưừửữứựỳỷỹýỵđÀẢÃÁẠĂẰẲẴẮẶÂẦẨẪẤẬÈẺẼÉẸÊỀỂỄẾỆÌỈĨÍỊÒỎÕÓỌÔỒỔỖỐỘƠỜỞỠỚỢÙỦŨÚỤƯỪỬỮỨỰỲỶỸÝỴĐ'

class Recognizor(object):
    'Recognize words image, output string'

    def __init__(self, nameLayerInput = "the_input", nameLayerPredict="softmax", \
                pathToModelJson="model.json", pathToWeightModel="weights19.h5"):

        self.__layerInput = nameLayerInput
        self.__layerPredict = nameLayerPredict
        self.__pathToWeightModel = pathToWeightModel

        json_file = open(pathToModelJson, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)

        model.load_weights(pathToWeightModel)
        print ("Loaded model !")
        self.__predictLayerModel = Model(inputs=model.get_layer(self.__layerInput).input, \
            outputs=model.get_layer(self.__layerPredict).output)

    def readWord(self, wordImage):

        img = cv2.resize(wordImage, (128, 64)).T
        tmp = img/255.0
        img = np.reshape(tmp, [1, 128, 64, 1])
        predicted = self.__predictLayerModel.predict(img) 
        out = self.decodeBatch(predicted)

        return out

    def labelsToText(self, labels):
        ret = [] 

        for c in labels:
            if c == len(alphabet): #CTC blank 
                ret.append("")
            else:
                ret.append(alphabet[c])
        return "".join(ret)

    def decodeBatch(self, out):

        out_best = list(np.argmax(out[0, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = self.labelsToText(out_best)
        return outstr



if __name__ == "__main__":
    recog = Recognizor()
    img = cv2.imread("tuan51.png", 0)
    print recog.readWord(img)