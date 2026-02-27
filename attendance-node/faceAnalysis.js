/**
 * FaceAnalysis - Node.js port of InsightFace's FaceAnalysis pipeline.
 *
 * Uses the same ONNX models as the Python version:
 *   - det_10g.onnx  (RetinaFace detection  – buffalo_l)
 *   - w600k_r50.onnx (ArcFace recognition – buffalo_l)
 *
 * Models are loaded from ~/.insightface/models/buffalo_l/ by default
 * (same path the Python insightface library downloads to).
 */

const ort = require('onnxruntime-node');
const sharp = require('sharp');
const path = require('path');
const os = require('os');

// ────────────────────────────────────────────────────────────────────
// ArcFace reference landmarks for a 112×112 aligned face crop.
// These are the 5-point destination coordinates used by InsightFace.
// ────────────────────────────────────────────────────────────────────
const ARCFACE_DST = [
  [38.2946, 51.6963],
  [73.5318, 51.5014],
  [56.0252, 71.7366],
  [41.5493, 92.3655],
  [70.7299, 92.2041],
];

// ════════════════════════════════════════════════════════════════════
//  MATH / IMAGE UTILITIES
// ════════════════════════════════════════════════════════════════════

/**
 * Convert HWC-RGB uint8 pixel data to an NCHW float32 blob.
 *   blob[c, h, w] = (pixel[h, w, c] – mean) * scaleFactor
 */
function createImageBlob(rgbData, width, height, mean, scaleFactor) {
  const size = 3 * height * width;
  const blob = new Float32Array(size);
  for (let c = 0; c < 3; c++) {
    const chOffset = c * height * width;
    for (let h = 0; h < height; h++) {
      const rowOffset = h * width;
      for (let w = 0; w < width; w++) {
        blob[chOffset + rowOffset + w] =
          (rgbData[(rowOffset + w) * 3 + c] - mean) * scaleFactor;
      }
    }
  }
  return blob;
}

/**
 * Decode bounding-box offsets from anchor centres.
 *   x1 = cx – left,   y1 = cy – top,
 *   x2 = cx + right,  y2 = cy + bottom
 *
 * @param {number} cx  Anchor centre x
 * @param {number} cy  Anchor centre y
 * @param {Float32Array} data  Raw bbox output (flat)
 * @param {number} offset  Starting index in data for this anchor
 * @param {number} stride  Feature-map stride
 * @returns {number[]} [x1, y1, x2, y2]
 */
function decodeBbox(cx, cy, data, offset, stride) {
  return [
    cx - data[offset + 0] * stride,
    cy - data[offset + 1] * stride,
    cx + data[offset + 2] * stride,
    cy + data[offset + 3] * stride,
  ];
}

/**
 * Decode 5 facial-landmark key-points from anchor centres.
 *
 * @returns {number[][]} [[x,y], …] length-5
 */
function decodeKps(cx, cy, data, offset, stride) {
  const kps = [];
  for (let j = 0; j < 10; j += 2) {
    kps.push([
      cx + data[offset + j] * stride,
      cy + data[offset + j + 1] * stride,
    ]);
  }
  return kps;
}

/**
 * Non-Maximum Suppression (NMS) – identical logic to InsightFace Python.
 *
 * @param {number[][]} dets  Each element is [x1, y1, x2, y2, score]
 * @param {number} thresh  IoU threshold
 * @returns {number[]}  Indices to keep
 */
function nms(dets, thresh) {
  if (dets.length === 0) return [];

  const x1 = dets.map((d) => d[0]);
  const y1 = dets.map((d) => d[1]);
  const x2 = dets.map((d) => d[2]);
  const y2 = dets.map((d) => d[3]);
  const scores = dets.map((d) => d[4]);
  const areas = dets.map(
    (_, i) => (x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1)
  );

  // Sort indices by score descending
  let order = scores
    .map((_, i) => i)
    .sort((a, b) => scores[b] - scores[a]);

  const keep = [];

  while (order.length > 0) {
    const i = order[0];
    keep.push(i);

    const remaining = [];
    for (let k = 1; k < order.length; k++) {
      const j = order[k];
      const xx1 = Math.max(x1[i], x1[j]);
      const yy1 = Math.max(y1[i], y1[j]);
      const xx2 = Math.min(x2[i], x2[j]);
      const yy2 = Math.min(y2[i], y2[j]);

      const w = Math.max(0, xx2 - xx1 + 1);
      const h = Math.max(0, yy2 - yy1 + 1);
      const inter = w * h;
      const ovr = inter / (areas[i] + areas[j] - inter);

      if (ovr <= thresh) remaining.push(j);
    }
    order = remaining;
  }

  return keep;
}

// ────────────────────────────────────────────────────────────────────
// Similarity-transform estimation  (replaces skimage SimilarityTransform)
// ────────────────────────────────────────────────────────────────────

/**
 * Estimate the 2×3 affine matrix for a similarity transform
 * that maps `src` landmarks → `dst` landmarks.
 *
 * Model:  dst = [[a, -b, tx], [b, a, ty]] · [src_x, src_y, 1]ᵀ
 * Solved via normal equations (A'A x = A'b).
 */
function estimateSimilarityTransform(src, dst) {
  const n = src.length;
  // Build A'A (4×4) and A'b (4×1) incrementally
  const ata = Array.from({ length: 4 }, () => new Float64Array(4));
  const atb = new Float64Array(4);

  for (let i = 0; i < n; i++) {
    const sx = src[i][0],
      sy = src[i][1];
    const dx = dst[i][0],
      dy = dst[i][1];

    //  Row 1: [sx, -sy, 1, 0]  →  dx
    //  Row 2: [sy,  sx, 0, 1]  →  dy
    const r1 = [sx, -sy, 1, 0];
    const r2 = [sy, sx, 0, 1];

    for (let j = 0; j < 4; j++) {
      for (let k = 0; k < 4; k++) {
        ata[j][k] += r1[j] * r1[k] + r2[j] * r2[k];
      }
      atb[j] += r1[j] * dx + r2[j] * dy;
    }
  }

  const x = solveLinear4x4(ata, atb);
  const a = x[0],
    b = x[1],
    tx = x[2],
    ty = x[3];
  return [
    [a, -b, tx],
    [b, a, ty],
  ];
}

/** Gaussian elimination with partial pivoting for a 4×4 system. */
function solveLinear4x4(A, B) {
  const n = 4;
  const a = A.map((row) => Float64Array.from(row));
  const b = Float64Array.from(B);

  for (let col = 0; col < n; col++) {
    // Pivot
    let maxVal = Math.abs(a[col][col]),
      maxRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(a[row][col]) > maxVal) {
        maxVal = Math.abs(a[row][col]);
        maxRow = row;
      }
    }
    [a[col], a[maxRow]] = [a[maxRow], a[col]];
    [b[col], b[maxRow]] = [b[maxRow], b[col]];

    // Eliminate
    for (let row = col + 1; row < n; row++) {
      const f = a[row][col] / a[col][col];
      for (let k = col; k < n; k++) a[row][k] -= f * a[col][k];
      b[row] -= f * b[col];
    }
  }

  // Back-substitution
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = b[i];
    for (let j = i + 1; j < n; j++) x[i] -= a[i][j] * x[j];
    x[i] /= a[i][i];
  }
  return x;
}

/**
 * Compute the 2×3 alignment matrix that warps detected landmarks
 * onto the ArcFace reference template (for a given image size).
 */
function estimateNorm(landmarks, imageSize = 112) {
  const ratio = imageSize / 112.0;
  const dst = ARCFACE_DST.map((p) => [p[0] * ratio, p[1] * ratio]);
  return estimateSimilarityTransform(landmarks, dst);
}

// ────────────────────────────────────────────────────────────────────
// warpAffine – bilinear interpolation  (matches cv2.warpAffine)
// ────────────────────────────────────────────────────────────────────

/**
 * Apply an affine warp with bilinear interpolation.
 *
 * `M` is the *forward* transform (src landmarks → dst template).
 * Internally we invert it so every dst pixel maps back to a src pixel.
 *
 * @param {Uint8Array|Buffer} srcData  HWC-RGB source pixel buffer
 * @param {number} srcW  Source width
 * @param {number} srcH  Source height
 * @param {number} channels  Channels (3)
 * @param {number[][]} M  2×3 affine matrix
 * @param {number} dstW  Output width
 * @param {number} dstH  Output height
 * @param {number} [borderValue=0]
 * @returns {Uint8Array}  HWC-RGB output
 */
function warpAffine(
  srcData,
  srcW,
  srcH,
  channels,
  M,
  dstW,
  dstH,
  borderValue = 0
) {
  const m00 = M[0][0],
    m01 = M[0][1],
    m02 = M[0][2];
  const m10 = M[1][0],
    m11 = M[1][1],
    m12 = M[1][2];

  // Invert the 2×3 affine matrix
  const det = m00 * m11 - m01 * m10;
  const id = 1.0 / det;
  const im00 = m11 * id;
  const im01 = -m01 * id;
  const im02 = (m01 * m12 - m11 * m02) * id;
  const im10 = -m10 * id;
  const im11 = m00 * id;
  const im12 = (m10 * m02 - m00 * m12) * id;

  const dst = new Uint8Array(dstW * dstH * channels);

  for (let dy = 0; dy < dstH; dy++) {
    for (let dx = 0; dx < dstW; dx++) {
      const sx = im00 * dx + im01 * dy + im02;
      const sy = im10 * dx + im11 * dy + im12;

      const x0 = Math.floor(sx);
      const y0 = Math.floor(sy);
      const x1 = x0 + 1;
      const y1 = y0 + 1;

      const fx = sx - x0;
      const fy = sy - y0;

      const w00 = (1 - fx) * (1 - fy);
      const w01 = fx * (1 - fy);
      const w10 = (1 - fx) * fy;
      const w11 = fx * fy;

      const dstIdx = (dy * dstW + dx) * channels;

      for (let c = 0; c < channels; c++) {
        let val = 0;

        if (x0 >= 0 && x0 < srcW && y0 >= 0 && y0 < srcH)
          val += w00 * srcData[(y0 * srcW + x0) * channels + c];
        else val += w00 * borderValue;

        if (x1 >= 0 && x1 < srcW && y0 >= 0 && y0 < srcH)
          val += w01 * srcData[(y0 * srcW + x1) * channels + c];
        else val += w01 * borderValue;

        if (x0 >= 0 && x0 < srcW && y1 >= 0 && y1 < srcH)
          val += w10 * srcData[(y1 * srcW + x0) * channels + c];
        else val += w10 * borderValue;

        if (x1 >= 0 && x1 < srcW && y1 >= 0 && y1 < srcH)
          val += w11 * srcData[(y1 * srcW + x1) * channels + c];
        else val += w11 * borderValue;

        dst[dstIdx + c] = Math.round(Math.min(255, Math.max(0, val)));
      }
    }
  }
  return dst;
}

// ════════════════════════════════════════════════════════════════════
//  FaceAnalysis  —  main public class
// ════════════════════════════════════════════════════════════════════

class FaceAnalysis {
  /**
   * @param {object} [options]
   * @param {string} [options.modelDir]  Path to the ONNX model directory.
   *   Defaults to ~/.insightface/models/buffalo_l
   * @param {number} [options.detThresh=0.5]  Detection confidence threshold
   * @param {number[]} [options.detSize=[640,640]]  Detection input [w, h]
   */
  constructor(options = {}) {
    this.modelDir =
      options.modelDir ||
      path.join(os.homedir(), '.insightface', 'models', 'buffalo_l');
    this.detThresh = options.detThresh ?? 0.5;
    this.nmsThresh = 0.4;
    this.detSize = options.detSize || [640, 640];

    // Auto-detected from the ONNX model during prepare()
    this.fmc = 3;
    this.featStrideFpn = [8, 16, 32];
    this.numAnchors = 2;
    this.useKps = true;

    // Preprocessing parameters (match Python InsightFace exactly)
    this.detInputMean = 127.5;
    this.detInputStd = 128.0;
    this.recInputMean = 127.5;
    this.recInputStd = 127.5;
    this.recInputSize = 112;

    // Caches
    this.centerCache = new Map();

    // Sessions (populated by prepare())
    this.detSession = null;
    this.recSession = null;
    this.detInputName = null;
    this.detOutputNames = null;
    this.recInputName = null;
    this.recOutputNames = null;
  }

  /** Load ONNX models into memory. Must be called once before `get()`. */
  async prepare() {
    const detPath = path.join(this.modelDir, 'det_10g.onnx');
    const recPath = path.join(this.modelDir, 'w600k_r50.onnx');

    console.log(`Loading detection model : ${detPath}`);
    this.detSession = await ort.InferenceSession.create(detPath);
    this.detInputName = this.detSession.inputNames[0];
    this.detOutputNames = [...this.detSession.outputNames];

    // Auto-detect model topology from the number of outputs
    const numOutputs = this.detOutputNames.length;
    if (numOutputs === 9) {
      this.fmc = 3;
      this.featStrideFpn = [8, 16, 32];
      this.numAnchors = 2;
      this.useKps = true;
    } else if (numOutputs === 15) {
      this.fmc = 5;
      this.featStrideFpn = [8, 16, 32, 64, 128];
      this.numAnchors = 1;
      this.useKps = true;
    } else if (numOutputs === 10) {
      this.fmc = 5;
      this.featStrideFpn = [8, 16, 32, 64, 128];
      this.numAnchors = 1;
      this.useKps = false;
    } else if (numOutputs === 6) {
      this.fmc = 3;
      this.featStrideFpn = [8, 16, 32];
      this.numAnchors = 2;
      this.useKps = false;
    }
    console.log(
      `  outputs=${numOutputs}  fmc=${this.fmc}  kps=${this.useKps}`
    );

    console.log(`Loading recognition model: ${recPath}`);
    this.recSession = await ort.InferenceSession.create(recPath);
    this.recInputName = this.recSession.inputNames[0];
    this.recOutputNames = [...this.recSession.outputNames];
    console.log('Models loaded successfully.\n');
  }

  // ──────────────────────────────────────────────────────────────────
  //  PUBLIC: get(imageBuffer) → [{bbox, score, kps, embedding}, …]
  // ──────────────────────────────────────────────────────────────────

  /**
   * Run full face-analysis pipeline on an image buffer (JPEG / PNG / etc.).
   *
   * @param {Buffer} imageBuffer  Raw file contents
   * @returns {Promise<Array<{bbox:number[], score:number, kps:number[][], embedding:number[]}>>}
   */
  async get(imageBuffer) {
    // Decode to raw RGB pixels
    const { data: rgbData, info } = await sharp(imageBuffer)
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });

    const imgW = info.width;
    const imgH = info.height;

    // 1.  Detect faces
    const { bboxes, kpss } = await this._detect(rgbData, imgW, imgH);
    if (bboxes.length === 0) return [];

    // 2.  Extract embedding for each face
    const faces = [];
    for (let i = 0; i < bboxes.length; i++) {
      const bbox = bboxes[i].slice(0, 4);
      const score = bboxes[i][4];
      const kps = kpss ? kpss[i] : null;

      let embedding = null;
      if (kps) {
        embedding = await this._getEmbedding(rgbData, imgW, imgH, kps);
      }
      faces.push({ bbox, score, kps, embedding });
    }
    return faces;
  }

  // ──────────────────────────────────────────────────────────────────
  //  PRIVATE: RetinaFace detection
  // ──────────────────────────────────────────────────────────────────

  async _detect(rgbData, imgW, imgH) {
    const [detW, detH] = this.detSize;

    // ── Letterbox resize ──
    const imRatio = imgH / imgW;
    const modelRatio = detH / detW;
    let newW, newH;
    if (imRatio > modelRatio) {
      newH = detH;
      newW = Math.floor(newH / imRatio);
    } else {
      newW = detW;
      newH = Math.floor(newW * imRatio);
    }
    const detScale = newH / imgH;

    const resizedBuf = await sharp(Buffer.from(rgbData), {
      raw: { width: imgW, height: imgH, channels: 3 },
    })
      .resize(newW, newH)
      .raw()
      .toBuffer();

    // Paste resized image onto a black (zero-padded) canvas
    const detImg = new Uint8Array(detW * detH * 3); // zeros
    for (let y = 0; y < newH; y++) {
      const srcOff = y * newW * 3;
      const dstOff = y * detW * 3;
      detImg.set(resizedBuf.subarray(srcOff, srcOff + newW * 3), dstOff);
    }

    // ── Create NCHW blob ──
    const blob = createImageBlob(
      detImg,
      detW,
      detH,
      this.detInputMean,
      1.0 / this.detInputStd
    );

    const inputTensor = new ort.Tensor('float32', blob, [1, 3, detH, detW]);
    const results = await this.detSession.run({
      [this.detInputName]: inputTensor,
    });

    // Ordered output tensors
    const netOuts = this.detOutputNames.map((name) => results[name]);

    // ── Decode detections per stride ──
    const allScores = [];
    const allBboxes = [];
    const allKpss = [];

    for (let idx = 0; idx < this.featStrideFpn.length; idx++) {
      const stride = this.featStrideFpn[idx];
      const scoresData = netOuts[idx].data;
      const bboxData = netOuts[idx + this.fmc].data;
      const totalAnchors = netOuts[idx].dims[0];

      const fmH = Math.floor(detH / stride);
      const fmW = Math.floor(detW / stride);

      // Build / retrieve anchor centres
      const cacheKey = `${fmH}_${fmW}_${stride}`;
      let anchors;
      if (this.centerCache.has(cacheKey)) {
        anchors = this.centerCache.get(cacheKey);
      } else {
        anchors = new Float32Array(fmH * fmW * this.numAnchors * 2);
        let ai = 0;
        for (let h = 0; h < fmH; h++) {
          for (let w = 0; w < fmW; w++) {
            for (let a = 0; a < this.numAnchors; a++) {
              anchors[ai++] = w * stride; // cx
              anchors[ai++] = h * stride; // cy
            }
          }
        }
        this.centerCache.set(cacheKey, anchors);
      }

      const kpsData = this.useKps
        ? netOuts[idx + this.fmc * 2].data
        : null;

      for (let i = 0; i < totalAnchors; i++) {
        const score = scoresData[i];
        if (score < this.detThresh) continue;

        const cx = anchors[i * 2];
        const cy = anchors[i * 2 + 1];

        const bbox = decodeBbox(cx, cy, bboxData, i * 4, stride);
        // Map back to original image coordinates
        allScores.push(score);
        allBboxes.push([
          bbox[0] / detScale,
          bbox[1] / detScale,
          bbox[2] / detScale,
          bbox[3] / detScale,
        ]);

        if (this.useKps && kpsData) {
          const kps = decodeKps(cx, cy, kpsData, i * 10, stride);
          allKpss.push(kps.map((p) => [p[0] / detScale, p[1] / detScale]));
        }
      }
    }

    if (allScores.length === 0) return { bboxes: [], kpss: [] };

    // ── Sort by score then NMS ──
    const dets = allBboxes.map((bb, i) => [...bb, allScores[i]]);
    const order = allScores
      .map((_, i) => i)
      .sort((a, b) => allScores[b] - allScores[a]);

    const sortedDets = order.map((i) => dets[i]);
    const sortedKpss = this.useKps ? order.map((i) => allKpss[i]) : null;

    const keep = nms(sortedDets, this.nmsThresh);

    return {
      bboxes: keep.map((i) => sortedDets[i]),
      kpss: this.useKps ? keep.map((i) => sortedKpss[i]) : null,
    };
  }

  // ──────────────────────────────────────────────────────────────────
  //  PRIVATE: ArcFace embedding extraction
  // ──────────────────────────────────────────────────────────────────

  async _getEmbedding(rgbData, imgW, imgH, kps) {
    // 1. Compute the alignment matrix (src landmarks → ArcFace template)
    const M = estimateNorm(kps, this.recInputSize);

    // 2. Warp the face into a 112×112 aligned crop
    const aligned = warpAffine(
      rgbData,
      imgW,
      imgH,
      3,
      M,
      this.recInputSize,
      this.recInputSize
    );

    // 3. Create NCHW blob  (pixel – 127.5) / 127.5
    const blob = createImageBlob(
      aligned,
      this.recInputSize,
      this.recInputSize,
      this.recInputMean,
      1.0 / this.recInputStd
    );

    // 4. Run ArcFace
    const inputTensor = new ort.Tensor('float32', blob, [
      1,
      3,
      this.recInputSize,
      this.recInputSize,
    ]);
    const results = await this.recSession.run({
      [this.recInputName]: inputTensor,
    });

    return Array.from(results[this.recOutputNames[0]].data);
  }
}

module.exports = FaceAnalysis;
