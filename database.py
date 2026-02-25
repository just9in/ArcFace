import insightface
import cv2
import os
import numpy as np
import pickle

app = insightface.app.FaceAnalysis()
app.prepare(ctx_id=0)

database = {}

for filename in os.listdir("known_face"):
    path = os.path.join("known_face", filename)
    img = cv2.imread(path)
    faces = app.get(img)

    if len(faces) > 0:
        embedding = faces[0].embedding
        name = os.path.splitext(filename)[0]
        database[name] = embedding
        print(f"Added {name}")

# Save database
with open("face_db.pkl", "wb") as f:
    pickle.dump(database, f)

print("Database saved.")