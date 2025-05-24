// routes/UploadRoutes.js
const express = require("express");
const router = express.Router();
const multer = require("multer");
const path = require("path");
const { handleUpload } = require("../controllers/UploadController");

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "uploads/");
  },
  filename: function (req, file, cb) {
    const ext = path.extname(file.originalname);
    const rawTitle = req.body.title || "untitled";
    const safeTitle = rawTitle.trim().replace(/[^a-z0-9_\-]/gi, "_");
    cb(null, `${safeTitle}${ext}`);
  },
});
const upload = multer({ storage });

router.post("/upload", upload.single("video"), handleUpload);

module.exports = router;
