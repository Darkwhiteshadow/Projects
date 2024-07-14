import cv2

car_xml = 'vehicle_detection_opencv/cars.xml'

carcascade = cv2.CascadeClassifier(car_xml)

video = cv2.VideoCapture('vehicle_detection_opencv/dataset/video2.avi')

framesize = (video.get(3),video.get(4))


while True:

    ret,frame = video.read()

    if not ret:
        break;

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    boxex = carcascade.detectMultiScale(gray,1.1,1)

    for( x,y,w,h) in boxex:

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)

    cv2.imshow('frame',frame)

cv2.destroyAllWindows()
