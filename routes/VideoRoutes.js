const express = require("express");
const router = express.Router();
const multer = require("multer");
const path = require("path");
const { uploadVideo, getAllVideos, getVideoById } = require("../controllers/VideoController");

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, "uploads/"),
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname);
    const safeTitle = req.body.title?.replace(/[^a-z0-9_\-]/gi, "_") || "untitled";
    cb(null, `${safeTitle}${ext}`);
  },
});
const upload = multer({ storage });

// Updated route
router.post("/upload", upload.single("video"), uploadVideo);
router.get("/", getAllVideos);
router.get("/:id", getVideoById);

module.exports = router;
