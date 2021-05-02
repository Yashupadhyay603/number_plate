import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

import math

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from keras_preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
from keras.models import model_from_json


# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    '''
    # Calculate new histogram with desired range and show histogram 
    new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    plt.plot(hist)
    plt.plot(new_hist)
    plt.xlim([0,256])
    plt.show()
    '''

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)

def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thresh = cv2.copyMakeBorder(thresh, 10, 10, 10, 10, cv2.BORDER_CONSTANT,value=(255,255,255))
    # cv2.imshow("im",thresh)
    # cv2.waitKey(0)
    
    

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    dilate = cv2.dilate(thresh, kernel, iterations=5)
    # cv2.imshow("im",dilate)
    # cv2.waitKey(0)


    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    largestContour = contours[1]
    
    minAreaRect = cv2.minAreaRect(largestContour)
    angle = minAreaRect[-1]
    i=1
    

    minAreaRect = cv2.minAreaRect(largestContour)


    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    # if angle < -45:
    #     angle = 90 + angle
    
    return -1.0 * angle

# Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

# Deskew image
def deskew(cvImage):
    angle = getSkewAngle(cvImage)

    if(math.fabs(angle)>60):

        angle+=math.degrees(math.pi)/2
    
    return rotateImage(cvImage, -1.0 * angle)



def prediction(image):
    
    loaded_model=load_model('englishwinglish.hdf5')
    
    characters = '0,1,2,3,4,5,6,7,8,9,A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z'
    characters = characters.split(',')
    # image= cv2.resize(img,(32,32))
    image=cv2.GaussianBlur(image,(3,3),0)
    ret,image=cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel=np.ones((2,2),np.uint8)    
    image=cv2.erode(image,kernel,iterations=1)   
    # plt.imshow(image,cmap='gray')
    # plt.show()
    # image[0:3,:]=255
    

    

    image = img_to_array(image)
    
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=3)
    
    
    
    
    output = loaded_model.predict(image/255.0)
    output = output.reshape(36)
    predicted = np.argmax(output)
    label = characters[predicted]
    success = output[predicted] * 100
    # print(predicted)
    if label in ['Q','O']:
        label='0'
    # if label in ['L']:
    #     label='4'
    
    # print(label)
    
    return label, success,predicted



def predict(img):   


    # img = cv2.imread('download (1)._yolo_out_py.jpg')
    # print(img)
    

    # # convert to gray
    # cv2.imshow("befor",img)
    # cv2.waitKey(0)
    

    img=cv2.fastNlMeansDenoisingColored(img,h=3,hColor=3,templateWindowSize=7,searchWindowSize=21)
    img = cv2.bilateralFilter(img, 20, 75, 75)
    # cv2.imshow("after",img)
    # cv2.waitKey(0)
    img=deskew(img)
    img=cv2.resize(img,(250,100),interpolation=cv2.INTER_CUBIC)


    auto_result, alpha, beta = automatic_brightness_and_contrast(img)
    

    # cv2.imshow('auto_result', auto_result)
    # cv2.waitKey(0)
    img=auto_result

    # cv2.imshow("im",img)
    # cv2.waitKey(0)


    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    


    # threshold grayscale image to extract glare
    mask=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,19,5)
    kernel=np.ones((1,1),np.uint8)    
    mask=cv2.dilate(mask,kernel,iterations=1)   
    # cv2.imshow("imask",mask)
    # cv2.waitKey(0)
    # mask=mask[15:90,10:]
    # epsilon = 0.1*cv2.arcLength(cnt,True)
    # approx = cv2.approxPolyDP(cnt,epsilon,True)  
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    list=[]
    list_index=[]
    # cv2.drawContours(img,contours,-1,(255,0,0),2)
    for i in range(len(contours)):

        x,y,w,h = cv2.boundingRect(contours[i])
        
        if 50<cv2.contourArea(contours[i])<900 and 1.1<h/w<3:
            
            x,y,w,h = cv2.boundingRect(contours[i])
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            
            c=mask[y:y+h,x:x+w]
            if len(c)>0:

            
                list.append(c)
                list_index.append(x)
    
    
    list_index2=list_index.copy()
    list_index2.sort()
    final_index=[]
    
    for i in list_index2:
        for j in range(len(list_index)):
            
            if i==list_index[j]:
                
                final_index.append(j)
                break;


    # print(final_index)
    list_ans=[]
    cv2.imshow("img",img)
    cv2.waitKey(0)
    for i in final_index:
        k=list[i]
        
        
        
        
        # x=cv2.adaptiveThreshold(x,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,1,1)
        ret, im=cv2.threshold(k,200,255,cv2.THRESH_BINARY)
        k=cv2.resize(im,(28,28),interpolation=cv2.INTER_AREA)
        k=cv2.bitwise_not(k)
        # cv2.imshow("im",list[i])
        # cv2.waitKey(0)
        
        # cv2.imshow("im2",k)q
        label,s,p=prediction(k)
        list_ans.append(label)

    
        # cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return list_ans


def predict2(original_image):
    
    # STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = 416
    # images = cv2.imread("data/images/car.jpg")

    saved_model_loaded = tf.saved_model.load('./checkpoints/custom-416', tags=[tag_constants.SERVING])

    # loop through images in list and run Yolov4 model on each
    # for count, image_path in enumerate(images, 1):

    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.
    


    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)

    
    infer = saved_model_loaded.signatures['serving_default']
    batch_data = tf.constant(images_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    # run non max suppression on detections
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.50
    )

    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
    original_h, original_w, _ = original_image.shape
    bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
    # print(bboxes.shape)
    # hold all detection data in one variable
    pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]
    # print(pred_bbox.shape)
    # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

    # by default allow all classes in .names file
    allowed_classes = list(class_names.values())
    

    crop_path='./final'
    k=crop_objects(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox, crop_path, allowed_classes)
    # print(len(k))
    # cv2.imshow("k",k)
    # cv2.waitKey(0)

    answer_list=[]
    # # cv2.imwrite(FLAGS.output + 'detection' + str(count) + '.png', image)
    # for i in k:
    #     answer_list.append(predict(i))
    image = utils.draw_bbox(original_image, pred_bbox, False, allowed_classes=allowed_classes, read_plate = False)
    
    image = Image.fromarray(image.astype(np.uint8))
    # if not FLAGS.dont_show:
    image.show()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

    return answer_list

# from keras.models import load_model






def test():
    '''
    We will be using a similar template to test your code
    '''
    image_paths =  ['test_multipleCar/p1.png','test_multipleCar/p2.png','test_multipleCar/p3.png','test_multipleCar/p4.png','test_multipleCar/p5.png']
    final=[]
    for i,image_path in enumerate(image_paths):
        image = cv2.imread(image_path) # This input format wont change
        # print(image)
        answer = predict2(image) # a list is expected
        final.append(answer)
        print(answer)
        cv2.waitKey(0)

    print(final)
if __name__ == "__main__":
    test()



