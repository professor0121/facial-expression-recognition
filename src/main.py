import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# camera = cv2.VideoCapture(0)

# while True:
#     success, frame=camera.read()
    
#     if not success:
#         print("camera open nahi ho raha hai")
#         break
#     cv2.imshow("My webcam",frame)
    
#     if cv2.waitKey(1)&0xFF == ord("q"):
#         break
    
# camera.release()
# cv2.destroyAllWindows()

model_path ="models/face_landmarker.task"

base_options=python.BaseOptions(model_asset_path=model_path)

options=vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)

landmarker=vision.FaceLandmarker.create_from_options(options)

camera=cv2.VideoCapture(0)

while True:
    success,frame=camera.read()
    
    if not success:
        print("Camera is not working !")
        break
    #camera to grascale
    
    rgb_frame = cv2.cvtColor(
    frame,
    cv2.COLOR_BGR2RGB
    )
    
    #detect face
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
        )
    
    result = landmarker.detect(mp_image)
    # Draw rectangle around every detected face
    
    if result.face_landmarks:
        print("Face detected !")
    
    #display webcam
    
    cv2.imshow("face detaction",frame)
    
    if cv2.waitKey(1)&0xFF==ord("q"):
        break

camera.release()
cv2.destroyAllWindows()