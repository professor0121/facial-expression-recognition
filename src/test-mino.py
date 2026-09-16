import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# 1. Load model
# -----------------------------

model_path = "models/face_landmarker.task"

base_options = python.BaseOptions(
    model_asset_path=model_path
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options
)

landmarker = vision.FaceLandmarker.create_from_options(options)


# -----------------------------
# 2. Read image
# -----------------------------

image = cv2.imread("data/face.jpg")

if image is None:
    print("Image nahi mili!")
    exit()


# -----------------------------
# 3. BGR → RGB
# -----------------------------

rgb_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# -----------------------------
# 4. OpenCV → MediaPipe Image
# -----------------------------

mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb_image
)


# -----------------------------
# 5. Detect landmarks
# -----------------------------

result = landmarker.detect(mp_image)


# -----------------------------
# 6. Print result
# -----------------------------

print("Number of faces:", len(result.face_landmarks))

if result.face_landmarks:

    face = result.face_landmarks[0]

    print("Number of landmarks:", len(face))

    print("First landmark:")
    print(face[0])