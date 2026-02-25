from flask import Flask, request, jsonify
import insightface
import numpy as np
import cv2

app = Flask(__name__)

model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=0)

@app.route("/generate-embedding", methods=["POST"])
def generate_embedding():

    file = request.files.get("image")

    if not file:
        return jsonify({"error": "No image provided"}), 400

    img = cv2.imdecode(
        np.frombuffer(file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    faces = model.get(img)

    if len(faces) != 1:
        return jsonify({"error": "Image must contain exactly one face"}), 400

    embedding = faces[0].embedding.tolist()

    return jsonify({
        "success": True,
        "embedding": embedding
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)