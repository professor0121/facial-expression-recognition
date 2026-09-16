import cv2
import mediapipe as mp
import math
from typing import Any

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
# Face mesh connections

MOUTH_OPEN_THRESHOLD = 0.30

FACE_CONNECTIONS = [
    # Face outline
    (10, 338), (338, 297), (297, 332), (332, 284),
    (284, 251), (251, 389), (389, 356), (356, 454),
    (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400),
    (400, 377), (377, 152),

    # Left eye
    (33, 133), (33, 159), (159, 158),
    (158, 157), (157, 173), (173, 133),
    (133, 155), (155, 154), (154, 153),
    (153, 145), (145, 144), (144, 163),
    (163, 7), (7, 33),

    # Right eye
    (362, 263), (362, 386), (386, 385),
    (385, 384), (384, 398), (398, 263),
    (263, 249), (249, 390), (390, 373),
    (373, 374), (374, 380), (380, 362),

    # Mouth
    (61, 146), (146, 91), (91, 181),
    (181, 84), (84, 17), (17, 314),
    (314, 405), (405, 321), (321, 375),
    (375, 291), (291, 61),

    # Nose
    (1, 2), (2, 98), (98, 327),
    (327, 326), (326, 2),
]
IMPORTANT_LANDMARKS = [
    1,      # Nose
    33,     # Left eye
    133,    # Left eye
    159,    # Left eye
    362,    # Right eye
    263,    # Right eye
    61,     # Left mouth
    291,    # Right mouth
    17,     # Mouth center
]

model_path ="models/face_landmarker.task"

def distance(point1, point2)->Any:
    
    return math.sqrt(
        (point1.x - point2.x) ** 2 +
        (point1.y - point2.y) ** 2
    )


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

    if result.face_landmarks:

        for face_landmarks in result.face_landmarks:

            height, width, _ = frame.shape

            for landmark in face_landmarks:

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    frame,
                    (x, y),
                    1,
                    (0, 255, 0),
                    -1
                )
                upper_lip = face_landmarks[13]
                lower_lip = face_landmarks[14]

                left_corner = face_landmarks[61]
                right_corner = face_landmarks[291]
                
                vertical_distance = distance(
                    upper_lip,
                    lower_lip
                )

                horizontal_distance = distance(
                    left_corner,
                    right_corner
                )

                mouth_opening_ratio = (
                    vertical_distance / horizontal_distance
                )
                
                if mouth_opening_ratio >= MOUTH_OPEN_THRESHOLD:
                    mouth_status = "MOUTH OPEN"
                else:
                    mouth_status = "MOUTH CLOSED"
                cv2.putText(
                    frame,
                    f"Mouth Ratio: {mouth_opening_ratio:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )
                                
            for index in IMPORTANT_LANDMARKS:
    
                landmark = face_landmarks[index]

                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.putText(
                    frame,
                    str(index),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1
                )
            for start_index, end_index in FACE_CONNECTIONS:
    
                start_landmark = face_landmarks[start_index]
                end_landmark = face_landmarks[end_index]

                start_x = int(start_landmark.x * width)
                start_y = int(start_landmark.y * height)

                end_x = int(end_landmark.x * width)
                end_y = int(end_landmark.y * height)

                cv2.line(
                    frame,
                    (start_x, start_y),
                    (end_x, end_y),
                    (0, 255, 0),
                    1
                )

    cv2.imshow("Face Mesh", frame)
        #display webcam
    
    if cv2.waitKey(1)&0xFF==ord("q"):
        break

camera.release()
cv2.destroyAllWindows()