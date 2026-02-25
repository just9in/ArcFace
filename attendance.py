from flask import Flask, request, jsonify
import insightface
import numpy as np
import cv2

app = Flask(__name__)       #api create 

model = insightface.app.FaceAnalysis()      #load models
model.prepare(ctx_id=0)
# model.prepare(ctx_id=0, det_size=(480, 480))  #for heavier images detection 4k

@app.route("/query-embedding", methods=["POST"])             #endpoint   
def extract_embeddings():

    file = request.files.get("image")

    if not file:
        return jsonify({"error": "No image provided"}), 400

    img = cv2.imdecode(                                     #converts to opencv image
        np.frombuffer(file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    faces = model.get(img)

    if len(faces) == 0:
        return jsonify({
            "success": False,
            "message": "No faces detected",
            "faces_detected": 0
        })

    result = []

    for face in faces:
        embedding = face.embedding
        embedding = embedding / np.linalg.norm(embedding)  # normalize

        bbox = face.bbox.astype(int).tolist()                   #to locate the face in image

        result.append({
            "bbox": bbox,
            "embedding": embedding.tolist()             #convert numpy to python list
        })

    return jsonify({
        "success": True,
        "faces_detected": len(result),
        "faces": result
    })

#run server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)