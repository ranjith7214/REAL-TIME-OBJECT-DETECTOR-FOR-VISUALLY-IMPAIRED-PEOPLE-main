import cv2
import tkinter as tk
from tkinter import *
import cv2
import pyttsx3

import time
from time import sleep

root=Tk()
root.geometry('612x612')
imb = PhotoImage(file="1.png")
label = Label(root,image=imb)
label.place(x=0,y=0)

l=Label(root,text='OBJECT DETECTION FOR BLIND PEOPLE',font=('times',16))
l.place(x=120,y=15)

def camera():
    cam = cv2.VideoCapture(0)

    while True:
        check, frame = cam.read()

        cv2.imshow('video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
def speak():
    engine = pyttsx3.init()


    classNames = []
    classFile = r"./coco.names"
    with open(classFile,"rt") as f:
        classNames = f.read().rstrip("\n").split("\n")
    
    configPath = r"./ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    weightsPath = r"./frozen_inference_graph.pb"
    
    net = cv2.dnn_DetectionModel(weightsPath,configPath)
    net.setInputSize(320,320)
    net.setInputScale(1.0/ 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    
    
    def getObjects(img, thres, nms, draw=True, objects=[]):
        classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
        #print(classIds,bbox)
        if len(objects) == 0: objects = classNames
        objectInfo =[]
        if len(classIds) != 0:
            for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
                className = classNames[classId - 1]
                print(className)
                engine.say(className)
                engine.runAndWait()
                
                if className in objects:
                    objectInfo.append([box,className])
                    if (draw):
                        cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                        cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                        cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    
        return img,objectInfo
    
    
    if __name__ == "__main__":
    
        cap = cv2.VideoCapture(0)
        cap.set(3,640)
        cap.set(4,480)
        #cap.set(10,70)
    
    
        while True:
            success, img = cap.read()
            result, objectInfo= getObjects(img,0.45,0.2)
            val=objectInfo[1]
            out=val[1]
            if out=='persone':
                len(objectInfo)
            else:
                pass
            print(len(objectInfo))
            #print(objectInfo)
            cv2.imshow("Output",img)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('a'):
                break
    cap.release()
    cv2.destroyAllWindows()
b1=Button(root,command=speak,text='CAMERA ON',fg='white',bg='black',font=('times'))
b1.place(x=250,y=230)


root.mainloop()