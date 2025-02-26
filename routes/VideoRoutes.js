const express = require("express");
const router = express.Router();
const { uploadVideo, getAllVideos, getVideoById } = require("../controllers/VideoController");

// Routes for Videos
router.post("/upload", uploadVideo);
router.get("/", getAllVideos);
router.get("/:id", getVideoById);

module.exports = router;
