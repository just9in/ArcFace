import cv2
import insightface
import numpy as np

# Load model
app = insightface.app.FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))

# Read image
img = cv2.imread("test.jpeg")

faces = app.get(img)

print("Faces detected:", len(faces))

for face in faces:
    print("Embedding shape:", face.embedding.shape)  # Should be (512,)

