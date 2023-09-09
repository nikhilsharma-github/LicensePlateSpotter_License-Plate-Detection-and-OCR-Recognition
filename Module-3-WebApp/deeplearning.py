#this is our main file which is using our saved deep learning model and using the pipeline function to perform the object detection and result presentation from this frontend platform


#this is the same pipeline function about which we have talked earlier in Model Prediction file in our jupyter notebook, we can also refer it from there.



#importing modules as numpy for numerical analysis, cv2 for computer vision library
# matplotlib for Data analysis and presentation
# tensorflow for deep learning
# image preprocessing libraries from tensorflow
# pytesseract for OCR of our licenseplate
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import pytesseract as pt


# loading our model from our local directory.
model = tf.keras.models.load_model('./static/models/object_detection.h5')


# in this pipeline function, which we have created earlier.
# providing the path of the image and filename as parameters in this function.
def object_detection(path,filename):
    #read image
    image=load_img(path) #PIL Object
    image=np.array(image,dtype=np.uint8) # 8 bit array (0,255)
    image1=load_img(path,target_size=(224,224))
    #data preprocessing
    image_arr_224=img_to_array(image1)/255.0 #convert into array and get the normalized output
    h,w,d=image.shape
    test_arr=image_arr_224.reshape(1,224,224,3)

    #make Predictions
    coords=model.predict(test_arr)

    #Denormalize
    denorm=np.array([w,w,h,h])
    coords=coords*denorm

    #Changing the Datatype
    coords=coords.astype(np.int32)

    # Drawing Bounding box on the top of the image
    xmin,xmax,ymin,ymax=coords[0]
    pt1=(xmin,ymin)
    pt2=(xmax,ymax)
    print(pt1,pt2)
    cv2.rectangle(image,pt1,pt2,(0,255,0),3)
    
    #convert image into bgr format
    image_bgr=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    cv2.imwrite('./static/predict/{}'.format(filename),image_bgr)
    return coords



# this is the OCR function which takes the image and its filepath, and returns the detected text in the form of a string value.
# this value is being obtained in app.py file, from there it is being observed and shown to our frontend platform.
def OCR(path,filename):
        img = np.array(load_img(path))
        coords = object_detection(path,filename)
        xmin,xmax,ymin,ymax=coords[0]
        roi = img[ymin:ymax,xmin:xmax]
        roi_bgr=cv2.cvtColor(roi,cv2.COLOR_RGB2BGR)
        cv2.imwrite('./static/roi/{}'.format(filename),roi_bgr)
        text=pt.image_to_string(roi)
        print(len(text))
        print(text)
        return text