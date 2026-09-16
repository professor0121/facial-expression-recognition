import cv2

camera = cv2.VideoCapture(0)

while True:
    success, frame=camera.read()
    
    if not success:
        print("camera open nahi ho raha hai")
        break
    cv2.imshow("My webcam",frame)
    
    if cv2.waitKey(1)&0xFF == ord("q"):
        break
    
camera.release()
cv2.destroyAllWindows()