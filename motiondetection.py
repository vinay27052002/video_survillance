import cv2 as cv 
import time 

video=cv.VideoCapture(0)
first_frame=None

while True:
    check,frame=video.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    gray=cv.GaussianBlur(gray,(21,21),0)#blurring the image for smoothing process for smooth detection
    #making first frame as reference freame for motion detection 
    if first_frame is None:
        first_frame=gray
        continue
    delta_frame=cv.absdiff(first_frame,gray)
    #making thrishold varable for avoiding noice detection in video
    thrishold_frame=cv.threshold(delta_frame,50,255,cv.THRESH_BINARY)[1]
    thrishold_frame=cv.dilate(thrishold_frame,None,iterations=2)

    #countour detection
    (cntr,_)=cv.findContours(thrishold_frame.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)

    for contour in cntr:
        if cv.contourArea(contour)<1000:#contour area specification
            continue
        (x,y,w,h)=cv.boundingRect(contour)
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)


    cv.imshow('video',frame)
    key=cv.waitKey(1)
    if key==ord("q"):
        break
#calling the functions 
video.release()
cv.destroyAllWindows()   