import cv2
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf
import keras

model = keras.models.load_model(r"D:\CV Project\Sign Language Recognition\code\model2.h5")

word_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
             10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
             20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}
color_dict = (0, 255, 0)
x = 0
y = 0
w = 64
h = 64

img_size = 128
minValue = 70
source = cv2.VideoCapture(1)
count = 0
string = " "
pred = " "
prev_val = 0

while True:
    ret, img = source.read()
    img = cv2.flip(img, 1)
    ret = cv2.flip(ret, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.rectangle(img,(x,y),(x+w,y+h),color_dict,2)
    cv2.rectangle(img, (24, 24), (250, 250), color_dict, 2)
    crop_img = gray[24:250, 24:250]
    count = count + 1
    if count % 25 == 0:
        prev_val = count
    # cv2.putText(img, str(prev_val // 100), (300, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    blur = cv2.GaussianBlur(crop_img, (5, 5), 2)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    resized = cv2.resize(res, (img_size, img_size))
    normalized = resized / 255.0
    reshaped = np.reshape(normalized, (1, img_size, img_size, 1))
    result = model.predict(reshaped)
    # print(result)
    label = np.argmax(result, axis=1)[0]
    pred = word_dict[label]
    if count == 50:
        count = 24
        pred = word_dict[label]
        if label == 0:
            string = string + " "
        # if(len(string)==1 or string[len(string)] != " "):

        else:
            string = string + pred

    cv2.putText(img, pred, (24, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    # cv2.rectangle(img, (275, 275), (70 , 300), (0, 0, 0), -1)
    cv2.putText(img, string, (275, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Gray", res)
    cv2.imshow('LIVE', img)
    key = cv2.waitKey(1)

    if key == 27:  # press Esc. to exit
        break
print(string)
cv2.destroyAllWindows()
source.release()

cv2.destroyAllWindows()
