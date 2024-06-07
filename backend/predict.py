from ultralytics import YOLO
import cv2
import matplotlib.pyplot as ptl
from keras.utils import load_img,img_to_array
import numpy as np
import pytesseract as pt
import os
import time
import pickle
#path='./liscense_plate_detectionreact/backend/testimages/Cars0.png'
model=YOLO('D:/project/liscense_plate_detectionreact/runs/detect/train5/weights/best.pt')
model_ocr=YOLO('D:/project/liscense_plate_detectionreact/runs/detect/train_ocr/weights/best.pt')
ocr={
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: 'A',
    11: 'B',
    12: 'C',
    13: 'D',
    14: 'E',
    15: 'F',
    16: 'G',
    17: 'H',
    18: 'I',
    19: 'J',
    20: 'K',
    21: 'L',
    22: 'M',
    23: 'N',
    24: 'O',
    25: 'P',
    26: 'Q',
    27: 'R',
    28: 'S',
    29: 'T',
    30: 'U',
    31: 'V',
    32: 'W',
    33: 'X',
    34: 'Y',
    35: 'Z'
}
def OCR(path,filename):
    image_save=load_img(path)
    image_save=np.array(image_save,dtype=np.uint8)
    cv2.imwrite('D:/project/liscense_plate_detectionreact/backend/static/upload/'+filename,image_save)
    results=model.predict(source=path,conf=0.3)
    boxes=results[0].boxes
    class_plate=boxes.cls
    coord=boxes.xyxy
    text=[]
    # height=boxes.xywh
    n_coord=len(coord)
    

    for i in range(0,n_coord):
        plate_class=int(class_plate[i])
        # h=int(height[i][3])
        min=(int(coord[i][0]),int(coord[i][1]))
        max=(int(coord[i][2]),int(coord[i][3]))
        xmin=min[0]
        ymin=min[1]
        xmax=max[0]
        ymax=max[1]
        image=load_img(path)
        image=np.array(image,dtype=np.uint8)
        image1=np.array(image,dtype=np.uint8)
        cv2.rectangle(image,min,max,(0,255,0),2)
        # UPLOAD_PATH_save=os.path.join(BASE_PATH,'WEBAPP/static/predict_numberplate/{}')

        cv2.imwrite('D:/project/liscense_plate_detectionreact/backend/static/predict_numberplate/'+str(i)+filename,image)
        # ptl.figure(figsize=(10,8))
        # ptl.imshow(image)
        # ptl.show()
        img_roi=image1[ymin:ymax,xmin:xmax]
        enlarge=cv2.resize(img_roi,None,fx=3.7,fy=3.7,interpolation=cv2.INTER_CUBIC)
        h=enlarge.shape[0]
    
    
        blur=cv2.GaussianBlur(enlarge,(5,5),0)
        sharpen_filter=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    # applying kernels to the input image to get the sharpened image
        sharp_image=cv2.filter2D(blur,-1,sharpen_filter)

    # # threshold the image, setting all foreground pixels to
    # # 255 and all background pixels to 0
        

        cv2.imwrite('D:/project/liscense_plate_detectionreact/backend/static/plates/'+str(i)+filename,sharp_image)
        result_ocr=model_ocr.predict(source=blur,conf=0.3)
        boxes_ocr=result_ocr[0].boxes
        data_ocr=boxes_ocr.data
        n=len(data_ocr)
        if(plate_class == 0):
            xaxis=[]
            class_name=[]
            for i in range (0,n):
                x=int(data_ocr[i][2])
                xaxis.append(x)
                y=int(data_ocr[i][5])
                class_name.append(y)
            for i in range(0,n-1):
                for j in range(i+1,n):
                    if(xaxis[i]>xaxis[j]):
                            t=xaxis[i]
                            xaxis[i]=xaxis[j]
                            xaxis[j]=t
                            t=class_name[i]
                            class_name[i]=class_name[j]
                            class_name[j]=t
            decode=""
            for i in range(0,n):
                decode=decode+(ocr.get(class_name[i]))
            print(decode)
            text.append(decode)
        else:
            xaxis=[]
            xaxis2=[]
            yaxis=[]
            yaxis2=[]
            label=[]
            label2=[]

            for i in range(0,n):
                x=int(data_ocr[i][0])
                y=int(data_ocr[i][3])
                l=int(data_ocr[i][5])
                if(y<(h/1.6)):
                    xaxis.append(x)
                    yaxis.append(y)
                    label.append(l)
                else:
                    xaxis2.append(x)
                    yaxis2.append(y)
                    label2.append(l)   
            for i in range(0,len(xaxis)-1):
                for j in range(i+1,len(xaxis)):
                    if(xaxis[i]>xaxis[j]):
                        temp=xaxis[i]
                        xaxis[i]=xaxis[j]
                        xaxis[j]=temp

                        temp=label[i]
                        label[i]=label[j]
                        label[j]=temp
            for i in range(0,len(xaxis2)-1):
                for j in range(i+1,len(xaxis2)):
                    if(xaxis2[i]>xaxis2[j]):
                        temp=xaxis2[i]
                        xaxis2[i]=xaxis2[j]
                        xaxis2[j]=temp

                        temp=label2[i]
                        label2[i]=label2[j]
                        label2[j]=temp
            numberPlate=""       
            for i in range(0,len(label)):
                numberPlate=numberPlate+(ocr.get(label[i]))
            for i in range(0,len(label2)):
                numberPlate=numberPlate+(ocr.get(label2[i]))
            print(numberPlate)
            text.append(numberPlate)
    return text

