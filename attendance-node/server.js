/**
 * attendance-node/server.js
 *
 * Express server with face registration & recognition:
 *
 *   POST /register        – register a face (exactly 1 face required)
 *   POST /recognize        – recognize faces in a group image
 *   GET  /registered       – list all registered names
 *   DELETE /registered/:name – remove a person from the database
 *   POST /query-embedding  – raw embedding extraction (original)
 *
 * Uses the same RetinaFace + ArcFace ONNX models from
 * ~/.insightface/models/buffalo_l/
 * Embeddings are persisted in face_db.pkl (JSON-encoded).
 */

const express = require('express');
const multer = require('multer');
const FaceAnalysis = require('./faceAnalysis');
const FaceDatabase = require('./faceDatabase');

const app = express();
const upload = multer({ storage: multer.memoryStorage() });

// ── Initialise the face-analysis model & database (loaded once at startup) ──
const model = new FaceAnalysis();
const faceDb = new FaceDatabase();

// ── POST /register ───────────────────────────────────────────────
// Registers a single face. The image MUST contain exactly 1 face.
// Form fields: "image" (file), "name" (text — the person's name).
app.post('/register', upload.single('image'), async (req, res) => {
  try {
    const name = req.body.name;
    if (!name) {
      return res.status(400).json({ error: 'Missing "name" field' });
    }
    if (!req.file) {
      return res.status(400).json({ error: 'No image provided' });
    }

    const faces = await model.get(req.file.buffer);

    if (faces.length === 0) {
      return res.status(400).json({ error: 'No face detected in the image' });
    }
    if (faces.length > 1) {
      return res.status(400).json({
        error: `Image must contain exactly 1 face, but ${faces.length} were detected`,
      });
    }

    faceDb.register(name, faces[0].embedding);

    return res.json({
      success: true,
      message: `Registered "${name}" successfully`,
      total_registered: faceDb.size,
    });
  } catch (err) {
    console.error('Error in /register:', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
});

// ── POST /recognize ──────────────────────────────────────────────
// Send a group image; returns recognised names for every detected face.
app.post('/recognize', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image provided' });
    }
    if (faceDb.size === 0) {
      return res.status(400).json({ error: 'Face database is empty — register faces first' });
    }

    const threshold = parseFloat(req.body.threshold) || 0.4;
    const faces = await model.get(req.file.buffer);

    if (faces.length === 0) {
      return res.json({
        success: false,
        message: 'No faces detected',
        faces_detected: 0,
        results: [],
      });
    }

    const results = faces.map((face) => {
      const { name, similarity } = faceDb.findMatch(face.embedding, threshold);
      return {
        bbox: face.bbox.map((v) => Math.floor(v)),
        name: name || 'Unknown',
        similarity: parseFloat(similarity.toFixed(4)),
      };
    });

    return res.json({
      success: true,
      faces_detected: results.length,
      results,
    });
  } catch (err) {
    console.error('Error in /recognize:', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
});

// ── GET /registered ──────────────────────────────────────────────
// List all registered persons.
app.get('/registered', (_req, res) => {
  return res.json({
    success: true,
    count: faceDb.size,
    names: faceDb.listNames(),
  });
});

// ── DELETE /registered/:name ─────────────────────────────────────
// Remove a person from the database.
app.delete('/registered/:name', (req, res) => {
  const { name } = req.params;
  const deleted = faceDb.remove(name);
  if (!deleted) {
    return res.status(404).json({ error: `"${name}" not found in database` });
  }
  return res.json({
    success: true,
    message: `Removed "${name}" from database`,
    total_registered: faceDb.size,
  });
});

// ── POST /query-embedding ────────────────────────────────────────
app.post('/query-embedding', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image provided' });
    }

    const faces = await model.get(req.file.buffer);

    if (faces.length === 0) {
      return res.json({
        success: false,
        message: 'No faces detected',
        faces_detected: 0,
      });
    }

    const result = faces.map((face) => {
      // L2-normalise the embedding (same as Python: embedding / np.linalg.norm)
      const norm = Math.sqrt(
        face.embedding.reduce((sum, v) => sum + v * v, 0)
      );
      const normalizedEmbedding = face.embedding.map((v) => v / norm);

      return {
        bbox: face.bbox.map((v) => Math.floor(v)),
        embedding: normalizedEmbedding,
      };
    });

    return res.json({
      success: true,
      faces_detected: result.length,
      faces: result,
    });
  } catch (err) {
    console.error('Error processing request:', err);
    return res.status(500).json({ error: 'Internal server error' });
  }
});

// ── Start ────────────────────────────────────────────────────────
async function main() {
  console.log('Preparing models…');
  await model.prepare();

  const PORT = process.env.PORT || 5001;
  app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running on http://0.0.0.0:${PORT}`);
  });
}

main().catch((err) => {
  console.error('Fatal:', err);
  process.exit(1);
});
