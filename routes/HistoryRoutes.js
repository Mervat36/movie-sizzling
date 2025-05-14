const express = require("express");
const router = express.Router();
const UserQuery = require("../models/UserQuery");
const { ensureAuthenticated } = require("../middleware/auth");

// GET /history - show user's search history with pagination
router.get("/history", ensureAuthenticated, async (req, res) => {
  try {
    const userId = req.session.user?._id || req.user?._id;
    if (!userId) {
      return res.render("history", {
        user: null,
        queries: [],
        currentPage: 1,
        totalPages: 1,
      });
    }

    const page = parseInt(req.query.page) || 1;
    const limit = 5;
    const skip = (page - 1) * limit;

    const totalCount = await UserQuery.countDocuments({ userId });
    const totalPages = Math.ceil(totalCount / limit);

    const rawQueries = await UserQuery.find({ userId })
      .sort({ timestamp: -1 })
      .skip(skip)
      .limit(limit)
      .lean();

    // ✅ Convert to expected format
    const queries = rawQueries.map((q) => {
      console.log("🟢 RAW ENTRY", q);
      const videos = [];

      // Main video
      if (q.resultVideoUrl) {
        videos.push({
          url: q.resultVideoUrl,
          caption: q.caption || "No caption",
          startTime: q.startTime || "",
          endTime: q.endTime || "",
        });
      }

      // Additional videos if any
      if (Array.isArray(q.additionalVideos)) {
        q.additionalVideos.forEach((v, idx) => {
          videos.push({
            url: v.url,
            caption: v.caption || `Result ${idx + 2}`,
            startTime: v.startTime || "",
            endTime: v.endTime || "",
          });
        });
      }

      return {
        query: q.query,
        videos,
      };
    });

    res.render("history", {
      user: req.session.user || req.user,
      queries,
      currentPage: page,
      totalPages,
    });
  } catch (err) {
    console.error("❌ Failed to retrieve search history:", err.message);
    res.render("history", {
      user: req.session.user || req.user,
      queries: [],
      currentPage: 1,
      totalPages: 1,
    });
  }
});

// POST /history/remove
router.post("/history/remove", ensureAuthenticated, async (req, res) => {
  const { videoUrl } = req.body;
  try {
    await UserQuery.deleteOne({
      resultVideoUrl: videoUrl,
      userId: req.session.user?._id || req.user?._id,
    });
    res.redirect("/history");
  } catch (err) {
    console.error("❌ Failed to delete video:", err.message);
    res.redirect("/history");
  }
});

module.exports = router;
