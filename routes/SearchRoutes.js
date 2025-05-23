const express = require("express");
const router = express.Router();
const UserQuery = require("../models/UserQuery");
router.get("/result", require("../controllers/ResultsController").showResult);
const { ensureAuthenticated } = require("../middleware/auth");
const path = require("path");
const { exec } = require("child_process");
const { searchEngine, saveSearchHistory } = require("../utils/searchEngine"); // create this module

// ✅ New: GET /search to support ?videoId=
router.get("/", ensureAuthenticated, async (req, res) => {
  const videoId = req.query.videoId;
  const videoTitle = videoId
    ? videoId.replace(/^dl_/, "").replace(/\.mp4$/, "")
    : "";

  res.render("search", {
    videoId,
    videoTitle,
  });
});

// POST: Handle form submission
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

// GET: Rerun search for an existing video
router.get("/search-again/:videoId", ensureAuthenticated, async (req, res) => {
  const videoId = req.params.videoId;
  const Video = require("../models/Video");

  const video = await Video.findById(videoId);
  if (!video) return res.redirect("/history");

  const safeTitle = video.filename.replace(/^dl_/, "").replace(/\.mp4$/, "");
  req.session.videoTitle = safeTitle;

  res.render("search", {
    videoId: videoId,
    videoTitle: safeTitle,
    videoFilename: video.filename
  });
});

module.exports = router;
