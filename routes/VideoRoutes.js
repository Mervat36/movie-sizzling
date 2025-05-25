const express = require("express");
const router = express.Router();
const multer = require("multer");
const path = require("path");
const { exec } = require("child_process");
const { ensureAuthenticated } = require("../middleware/auth");

const {
  uploadVideo,
  getAllVideos,
  getVideoById,
  deleteVideoByTitle,
} = require("../controllers/VideoController");

const { searchEngine, saveSearchHistory } = require("../utils/searchEngine");

// Multer Storage Config
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, "uploads/"),
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname);
    const safeTitle =
      req.body.title?.replace(/[^a-z0-9_\-]/gi, "_") || "untitled";
    cb(null, `${safeTitle}${ext}`);
  },
});
const upload = multer({ storage });

// === Routes ===

// Upload
router.post(
  "/upload",
  ensureAuthenticated,
  upload.single("video"),
  uploadVideo
);

// Search Result Generation
router.post("/result", ensureAuthenticated, async (req, res) => {
  const { query, videoId } = req.body;
  try {
    const result = await searchEngine(query, videoId);
    const userId = req.session.user?._id || "guest";
    await saveSearchHistory(userId, videoId, query, result);

    res.redirect(
      `/search/result?videoId=${videoId}&segment=${JSON.stringify(
        result.bestMatch
      )}&q=${encodeURIComponent(query)}`
    );
  } catch (error) {
    console.error("Search error:", error);
    res.status(500).send("Search error");
  }
});

// Render Result Clip
router.get("/result", async (req, res) => {
  const { videoId, segment } = req.query;
  const { start, end } = JSON.parse(segment);

  const inputPath = path.join(__dirname, "../uploads", `${videoId}.mp4`);
  const outputFile = `${videoId}_${start.replace(/:/g, "-")}_${end.replace(
    /:/g,
    "-"
  )}.mp4`;
  const outputPath = path.join(__dirname, "../output/clips", outputFile);

  const command = `ffmpeg -i "${inputPath}" -ss ${start} -to ${end} -c copy "${outputPath}"`;
  exec(command, (err) => {
    if (err) {
      console.error("FFmpeg error:", err);
      return res.status(500).send("Error generating clip");
    }
    res.render("results", { videoPath: `/output/clips/${outputFile}` });
  });
});

// Fetch video data
router.get("/", getAllVideos);
router.get("/:id", getVideoById);

// Delete a video and all data
router.delete("/delete/:title", ensureAuthenticated, deleteVideoByTitle);

module.exports = router;
