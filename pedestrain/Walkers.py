import cv2


vid=cv2.VideoCapture("walking.avi")

classifier=cv2.CascadeClassifier("haarcascade_fullbody.xml")


while True:
    ret,frame=vid.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    bodies=classifier.detectMultiScale(gray,1.2,3)
    for (x,y,w,h) in bodies:
      cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)
    cv2.imshow("video",frame)
    if cv2.waitKey(30)==32:
            break

vid.release()
cv2.destroyAllWindows()













