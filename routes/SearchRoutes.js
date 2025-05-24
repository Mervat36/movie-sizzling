const express = require("express");
const router = express.Router();
const UserQuery = require("../models/UserQuery");
const Video = require("../models/Video");
router.get("/result", require("../controllers/ResultsController").showResult);
const { ensureAuthenticated } = require("../middleware/auth");
const path = require("path");
const { exec } = require("child_process");
const { searchEngine, saveSearchHistory } = require("../utils/searchEngine");

router.get("/", ensureAuthenticated, async (req, res) => {
  const videoId = req.query.videoId;
  let videoFilename = null;
  let videoTitle = null;

  if (videoId) {
    try {
      const video = await Video.findById(videoId);
      if (video) {
        videoFilename = video.filename;
        videoTitle = video.title || video.filename;
      }
    } catch (err) {
      console.error("❌ Error fetching video:", err.message);
    }
  }

  res.render("search", {
    videoId,
    videoFilename: video?.filename || null,
    videoTitle: video?.title || video?.filename || null,
  });
});

// POST: Handle form submission
router.post("/result", ensureAuthenticated, async (req, res) => {
  const { query, videoId } = req.body;

  try {
    const video = await Video.findById(videoId);
    if (!video) {
      return res.status(404).render("error", {
        error: { status: 404, message: "Video not found" },
        theme: req.session.theme || "light",
        friendlyMessage: "The video associated with your search was not found.",
      });
    }

    const safeTitle = video.filename.replace(/^dl_/, "").replace(/\.mp4$/, "");
    req.session.videoTitle = safeTitle;

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

  const video = await Video.findById(videoId);
  if (!video) return res.redirect("/history");

  const resolvedTitle = video.title?.trim() || video.filename;
  req.session.videoTitle = resolvedTitle;

  res.render("search", {
    videoId,
    videoTitle: resolvedTitle,
    videoFilename: video.filename,
  });
});

module.exports = router;
