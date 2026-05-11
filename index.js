const express = require("express");
const bodyParser = require("body-parser");
const fs = require("fs");
const path = require("path");
const axios = require("axios");
const multer = require("multer");
const { execSync } = require("child_process");

const app = express();
app.use(bodyParser.json({ limit: "10mb" })); // Increase limit if needed
const uploadsDir = path.join(__dirname, "uploads");
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir);
}
const upload = multer({
  dest: uploadsDir,
  limits: { fileSize: 25 * 1024 * 1024 }, // 25MB cap for mp4
});

function getDefaultProxyBaseUrl() {
  // In many Windows+WSL2 setups, the WSL service is NOT reachable via 127.0.0.1.
  // Prefer the WSL VM IP when FLASK_URL is not explicitly set.
  try {
    if (process.platform === "win32") {
      const ip = String(execSync("wsl hostname -I", { stdio: ["ignore", "pipe", "ignore"] }))
        .trim()
        .split(/\s+/)[0];
      if (ip) return `http://${ip}:5000`;
    }
  } catch (_e) {
    // ignore and fall back
  }
  return "http://127.0.0.1:5000";
}

const PROXY_BASE_URL = process.env.FLASK_URL || getDefaultProxyBaseUrl();
const MAX_RETRIES = 3;
const TIMEOUT_MS = parseInt(process.env.OCR_TIMEOUT_MS || "120000", 10);
const MAX_BASE64_CHARS = parseInt(
  process.env.MAX_BASE64_CHARS || "5000000",
  10,
);

function maybeKeepBase64(value, fieldName) {
  if (typeof value !== "string") return null;
  if (value.length <= MAX_BASE64_CHARS) return value;
  console.warn(
    `Dropping oversized field ${fieldName} (${value.length} chars) to prevent mobile OOM`,
  );
  return null;
}

async function postWithRetry(url, data, config = {}) {
  let lastError = null;
  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      console.log(`Proxy request (attempt ${attempt}) -> ${url}`);
      return await axios.post(url, data, {
        timeout: TIMEOUT_MS,
        ...config,
      });
    } catch (error) {
      lastError = error;
      console.error(`Proxy attempt ${attempt} failed:`, error.message);
      if (error.response) {
        console.error("Proxy response status:", error.response.status);
        console.error("Proxy response data:", error.response.data);

        // Do not retry client errors (4xx). These are usually validation issues.
        if (error.response.status >= 400 && error.response.status < 500) {
          throw error;
        }
      }
      if (attempt < MAX_RETRIES) {
        const backoffMs = 1000 * Math.pow(2, attempt);
        await new Promise((resolve) => setTimeout(resolve, backoffMs));
      }
    }
  }
  throw lastError;
}

app.get("/health", (_req, res) => {
  res.json({ ok: true, proxyTo: PROXY_BASE_URL });
});

app.post("/ocr", upload.single("file"), async (req, res) => {
  try {
    const { box, photoBox, signatureBox, photoWidth, photoHeight } =
      req.body || {};
    let rawImagePath = null;

    if (req.file && req.file.path) {
      rawImagePath = req.file.path;
    } else if (req.body && req.body.image) {
      rawImagePath = path.join(__dirname, "uploads", `ktp_${Date.now()}.png`);
      fs.writeFileSync(rawImagePath, Buffer.from(req.body.image, "base64"));
    }

    if (!rawImagePath) {
      return res.status(400).json({ error: "No image/file provided" });
    }

    // Forward raw image to the local Flask OCR API with retries and extended timeout
    const OCR_URL = `${PROXY_BASE_URL}/ocr/ktp`;
    console.log(`Calling OCR backend -> ${OCR_URL}`);
    const FormData = require("form-data");
    const form = new FormData();
    form.append("file", fs.createReadStream(rawImagePath));
    if (box) {
      // Forward the exact box payload once (avoid double-stringifying JSON)
      form.append("box", typeof box === "string" ? box : JSON.stringify(box));
    }
    if (photoBox) {
      form.append(
        "photoBox",
        typeof photoBox === "string" ? photoBox : JSON.stringify(photoBox),
      );
    }
    if (signatureBox) {
      form.append(
        "signatureBox",
        typeof signatureBox === "string"
          ? signatureBox
          : JSON.stringify(signatureBox),
      );
    }
    if (photoWidth) form.append("photoWidth", String(photoWidth));
    if (photoHeight) form.append("photoHeight", String(photoHeight));

    const ocrResponse = await postWithRetry(OCR_URL, form, {
      headers: form.getHeaders(),
    });
    const {
      processed_image,
      ktp_face_base64,
      ktp_photo_base64,
      ktp_signature_base64,
      ...ktpData
    } = ocrResponse.data || {};

    // Store processed image (optional)
    if (processed_image) {
      const processedImagePath = path.join(
        uploadsDir,
        `ktp_processed_${Date.now()}.png`,
      );
      fs.writeFileSync(
        processedImagePath,
        Buffer.from(processed_image, "base64"),
      );
    }

    if (ktp_photo_base64) {
      const photoCropPath = path.join(
        uploadsDir,
        `ktp_photo_crop_${Date.now()}.png`,
      );
      fs.writeFileSync(photoCropPath, Buffer.from(ktp_photo_base64, "base64"));
    }
    if (ktp_signature_base64) {
      const sigCropPath = path.join(
        uploadsDir,
        `ktp_signature_crop_${Date.now()}.png`,
      );
      fs.writeFileSync(sigCropPath, Buffer.from(ktp_signature_base64, "base64"));
    }

    // Send only bounded image payloads to avoid OOM in React Native networking bridge.
    res.json({
      ...ktpData,
      processed_image: maybeKeepBase64(processed_image, "processed_image"),
      ktp_face_base64: maybeKeepBase64(ktp_face_base64, "ktp_face_base64"),
      ktp_photo_base64: maybeKeepBase64(ktp_photo_base64, "ktp_photo_base64"),
      ktp_signature_base64: maybeKeepBase64(
        ktp_signature_base64,
        "ktp_signature_base64",
      ),
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "Processing failed", details: err.message });
  }
});

app.post("/liveness", upload.single("video"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No video file provided" });
  }

  const filePath = req.file.path;
  try {
    const FormData = require("form-data");
    const form = new FormData();
    form.append("video", fs.createReadStream(filePath), {
      filename: req.file.originalname || "liveness.mp4",
      contentType: req.file.mimetype || "video/mp4",
    });
    if (req.body?.session_id) {
      form.append("session_id", String(req.body.session_id));
    }

    const response = await postWithRetry(`${PROXY_BASE_URL}/liveness`, form, {
      headers: form.getHeaders(),
      maxBodyLength: Infinity,
      maxContentLength: Infinity,
    });
    res.json(response.data);
  } catch (err) {
    console.error(err);
    const status = err.response?.status || 502;
    res.status(status).json({
      error: "Liveness detection failed",
      details: err.response?.data || err.message,
    });
  } finally {
    fs.unlink(filePath, () => {});
  }
});

app.post("/match/faces", async (req, res) => {
  try {
    const url = `${PROXY_BASE_URL}/match/faces`;
    const response = await postWithRetry(url, req.body, {
      headers: { "Content-Type": "application/json" },
    });
    res.json(response.data);
  } catch (err) {
    console.error(err);
    res.status(502).json({
      error: "Face matching failed after retries",
      details: err.message,
    });
  }
});

app.post("/match/signature", async (req, res) => {
  try {
    const url = `${PROXY_BASE_URL}/match/signature`;
    const response = await postWithRetry(url, req.body, {
      headers: { "Content-Type": "application/json" },
    });
    res.json(response.data);
  } catch (err) {
    console.error(err);
    const status = err.response?.status || 502;
    res.status(status).json({
      error: "Signature matching failed",
      details: err.response?.data || err.message,
    });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, "0.0.0.0", () => {
  console.log(`OCR backend running on port ${PORT}`);
});
