/**
 * faceDatabase.js
 *
 * Manages a face-embedding database stored as a .pkl file (binary JSON).
 * Each entry: { name: string, embedding: number[] (512-d, L2-normalised) }
 *
 * The .pkl file stores a JSON-serialised object:
 *   { "PersonName": [0.012, -0.034, ...], ... }
 *
 * This mirrors the Python database.py that uses pickle to store
 * { name: np.ndarray } — the on-disk format here is JSON-encoded binary
 * so the file can also be inspected / ported easily.
 */

const fs = require('fs');
const path = require('path');

const DEFAULT_DB_PATH = path.join(__dirname, 'face_db.pkl');

class FaceDatabase {
  /**
   * @param {string} [dbPath] — path to the .pkl database file
   */
  constructor(dbPath) {
    this.dbPath = dbPath || DEFAULT_DB_PATH;
    this.db = {}; // { name: number[] }
    this._load();
  }

  // ── persistence ─────────────────────────────────────────────────

  /** Load database from disk (no-op if file doesn't exist). */
  _load() {
    try {
      if (fs.existsSync(this.dbPath)) {
        const raw = fs.readFileSync(this.dbPath, 'utf-8');
        this.db = JSON.parse(raw);
        console.log(
          `Face database loaded: ${Object.keys(this.db).length} person(s) from ${this.dbPath}`
        );
      } else {
        console.log('No existing face database found — starting fresh.');
      }
    } catch (err) {
      console.error('Failed to load face database:', err.message);
      this.db = {};
    }
  }

  /** Persist database to disk. */
  _save() {
    fs.writeFileSync(this.dbPath, JSON.stringify(this.db), 'utf-8');
  }

  // ── public API ──────────────────────────────────────────────────

  /**
   * Register (or update) a person's face embedding.
   * @param {string} name
   * @param {number[]} embedding  — raw 512-d embedding (will be L2-normalised)
   */
  register(name, embedding) {
    this.db[name] = this._normalise(embedding);
    this._save();
  }

  /**
   * Remove a person from the database.
   * @returns {boolean} true if deleted, false if not found
   */
  remove(name) {
    if (!(name in this.db)) return false;
    delete this.db[name];
    this._save();
    return true;
  }

  /** List every registered name. */
  listNames() {
    return Object.keys(this.db);
  }

  /** Number of registered persons. */
  get size() {
    return Object.keys(this.db).length;
  }

  /**
   * Find the best match for a query embedding.
   *
   * @param {number[]} embedding — 512-d query embedding
   * @param {number}  [threshold=0.4] — minimum cosine similarity to accept
   * @returns {{ name: string|null, similarity: number }}
   */
  findMatch(embedding, threshold = 0.4) {
    const query = this._normalise(embedding);
    let bestName = null;
    let bestSim = -1;

    for (const [name, dbEmb] of Object.entries(this.db)) {
      const sim = this._cosine(query, dbEmb);
      if (sim > bestSim) {
        bestSim = sim;
        bestName = name;
      }
    }

    if (bestSim < threshold) {
      return { name: null, similarity: bestSim };
    }
    return { name: bestName, similarity: bestSim };
  }

  // ── helpers ─────────────────────────────────────────────────────

  /** L2-normalise a vector. */
  _normalise(vec) {
    const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
    if (norm === 0) return vec;
    return vec.map((v) => v / norm);
  }

  /** Cosine similarity between two L2-normalised vectors (= dot product). */
  _cosine(a, b) {
    let dot = 0;
    for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
    return dot;
  }
}

module.exports = FaceDatabase;
