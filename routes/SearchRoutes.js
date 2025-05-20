const express = require("express");
const router = express.Router();
const UserQuery = require("../models/UserQuery");
router.get("/result", require("../controllers/ResultsController").showResult);
const { ensureAuthenticated } = require("../middleware/auth");

// Route to handle POST request from form
const path = require("path");
const { exec } = require("child_process");
const { searchEngine, saveSearchHistory } = require("../utils/searchEngine"); // create this module

router.post("/result", ensureAuthenticated, async (req, res) => {
  const { query, videoId } = req.body;

  try {
    // Call your AI model or dummy logic for now
    const result = await searchEngine(query, videoId);

    // Save to history (optional: you can skip userId if not logged in)
    const userId = req.session.user?._id || "guest";
    await saveSearchHistory(userId, videoId, query, result);

    // Redirect to result page
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
router.get("/search-again/:videoId", ensureAuthenticated, async (req, res) => {
  const videoId = req.params.videoId;
  const Video = require("../models/Video");

  const video = await Video.findById(videoId);
  if (!video) return res.redirect("/history");

  const safeTitle = video.filename.replace(/^dl_/, "").replace(/\.mp4$/, "");
  req.session.videoTitle = safeTitle;

  res.render("search", {
    videoId: videoId,
  });
});

module.exports = router;
