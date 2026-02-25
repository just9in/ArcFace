import cv2
import insightface
import numpy as np
import os
import pickle

#start model    

app = insightface.app.FaceAnalysis()
app.prepare(ctx_id=0)

#db automation

def build_database():
    database = {}

    print("Scanning known_faces folder...")

    for filename in os.listdir("known_face"):
        if filename.endswith(".jpg") or filename.endswith(".png"):

            path = os.path.join("known_face", filename)
            img = cv2.imread(path)

            if img is None:
                continue

            faces = app.get(img)

            if len(faces) > 0:
                embedding = faces[0].embedding
                name = os.path.splitext(filename)[0]
                database[name] = embedding
                print(f"Loaded: {name}")

            else:
                print(f"No face detected in {filename}")

    # Save database
    with open("face_db.pkl", "wb") as f:
        pickle.dump(database, f)

    print("Database updated.\n")
    return database


# Build database automatically
database = build_database()

#cosine function

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

#webcam configs

video = cv2.VideoCapture(0)
print("Press 'q' to quit")

while True:
    ret, frame = video.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        embedding = face.embedding
        name = "Unknown"
        max_sim = 0

        for db_name, db_embedding in database.items():
            sim = cosine_similarity(embedding, db_embedding)

            if sim > max_sim and sim > 0.5:  # threshold
                max_sim = sim
                name = db_name

        box = face.bbox.astype(int)

        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(frame, name, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("ArcFace Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()