// 📁 controllers/historyController.js
const fs = require("fs");
const path = require("path");
const Video = require("../models/Video");
const ResultVideo = require("../models/ResultVideo");
const UserQuery = require("../models/UserQuery");
const ShotMetadata = require("../models/ShotData");
const SceneMetadata = require("../models/SceneMetadata");
const SceneResults = require("../models/SceneSearchResult");
const supabase = require("../supabaseClient");

exports.getHistoryPage = async (req, res) => {
  const userId = req.user._id;
  const page = parseInt(req.query.page) || 1;
  const limit = 3; // 3 videos per page
  const skip = (page - 1) * limit;

  const userQueries = await UserQuery.find({ userId });
  const searchedVideoIds = [...new Set(userQueries.map(q => q.videoId.toString()))];

  const totalVideos = searchedVideoIds.length;
  const totalPages = Math.ceil(totalVideos / limit);

  const videos = await Video.find(
    { _id: { $in: searchedVideoIds } },
    { isHidden: 1, title: 1, filename: 1, user: 1, createdAt: 1 }
  ).sort({ createdAt: 1 });


  const videoMap = {};

  await Promise.all(
    videos.map(async (video) => {
      const queries = await UserQuery.find({ videoId: video._id, userId });
      const queryMap = {};

      await Promise.all(
        queries.map(async (query) => {
          const results = await ResultVideo.find({ queryId: query._id });
          queryMap[query._id] = { query, results };
        })
      );

      videoMap[video._id] = { video, queries: queryMap };
    })
  );

  const toast = req.session.toast || null;
  delete req.session.toast;

  res.render("history", {
    videos,
    videoMap,
    toast,
    currentPage: page,
    totalPages,
    user: req.user,
  });
};

exports.downloadVideo = async (req, res) => {
  const video = await Video.findById(req.params.id);
  if (video.isHidden && video.user.toString() !== req.user._id.toString()) {
    return res.status(403).send("This video is private.");
  }
  const filePath = path.join(__dirname, "../uploads", video.filename);
  res.download(filePath);
};

exports.deleteVideo = async (req, res) => {
  try {
    const userId = req.user._id;
    const videoId = req.params.id;

    // Only delete user's query history related to this video
    const queries = await UserQuery.find({ userId, videoId });
    for (const query of queries) {
      await ResultVideo.deleteMany({ queryId: query._id });
    }
    await UserQuery.deleteMany({ userId, videoId });

    // Respond properly
    if (req.headers.accept?.includes("application/json")) {
      return res.status(200).json({ success: true });
    }

    req.session.toast = {
      type: "success",
      message: "Video removed from your history.",
    };
    res.redirect("/history");
  } catch (err) {
    console.error("❌ deleteVideo error:", err.message);
    if (req.headers.accept?.includes("application/json")) {
      return res.status(500).json({ success: false, message: "Internal error." });
    }
    req.session.toast = {
      type: "error",
      message: "Error deleting video from history.",
    };
    res.redirect("/history");
  }
};



exports.deleteQuery = async (req, res) => {
  const userId = req.user._id;
  const queryId = req.params.id;

  const query = await UserQuery.findOne({ _id: queryId, userId });
  if (!query) {
    return res.status(404).json({ success: false, message: "Query not found or not yours." });
  }

  await ResultVideo.deleteMany({ queryId });
  await UserQuery.findByIdAndDelete(queryId);

  if (req.headers.accept?.includes("application/json")) {
    return res.status(200).json({ success: true });
  }

  req.session.toast = {
    type: "success",
    message: "Query and results deleted.",
  };
  res.redirect("/history");
};


exports.downloadResult = async (req, res) => {
  const result = await ResultVideo.findById(req.params.id);
  const clipPath = path.join(
    __dirname,
    "../public/output/clips",
    result.clipFilename
  );
  res.download(clipPath);
};

exports.deleteResult = async (req, res) => {
  const userId = req.user._id;
  const result = await ResultVideo.findById(req.params.id);
  if (!result) {
    return res.status(404).json({ success: false, message: "Result not found." });
  }

  const query = await UserQuery.findOne({ _id: result.queryId, userId });
  if (!query) {
    return res.status(403).json({ success: false, message: "Not allowed to delete this result." });
  }

  // Now delete
  await ResultVideo.deleteMany({
    queryId: result.queryId,
    clipFilename: result.clipFilename,
  });

  const clipPath = path.join(__dirname, "../public/output/clips", result.clipFilename);
  if (fs.existsSync(clipPath)) fs.unlinkSync(clipPath);

  if (req.headers.accept?.includes("application/json")) {
    return res.status(200).json({ success: true });
  }

  req.session.toast = {
    type: "success",
    message: "Result deleted from this query.",
  };
  res.redirect("/history");
};
exports.renameVideo = async (req, res) => {
  const userId = req.user._id;
  const videoId = req.params.id;
  const newTitle = req.body.newTitle?.trim();
  const isHidden = String(req.body.isHidden).toLowerCase() === "true";

  if (!newTitle) {
    if (req.headers.accept?.includes("application/json")) {
      return res.status(400).json({ success: false, message: "Title cannot be empty." });
    }
    req.session.toast = { type: "error", message: "Title cannot be empty." };
    return res.redirect("/history");
  }

  try {
    const video = await Video.findById(videoId);
    if (!video) {
      return res.status(404).json({ success: false, message: "Video not found." });
    }

    const isOwner = video.user.toString() === userId.toString();

    // ✅ Update title in all UserQueries for this user
    const userQueries = await UserQuery.find({ userId, videoId });
    await Promise.all(
      userQueries.map(query =>
        UserQuery.findByIdAndUpdate(query._id, { customTitle: newTitle })
      )
    );

    // ✅ Only the owner can update visibility + real title
    if (isOwner) {
      video.isHidden = isHidden;
      video.title = newTitle;
      await video.save();
    }

    if (req.headers.accept?.includes("application/json")) {
      return res.status(200).json({ success: true });
    }

    req.session.toast = {
      type: "success",
      message: "Video renamed and visibility updated.",
    };
    res.redirect("/history");
  } catch (err) {
    console.error("Rename error:", err);
    if (req.headers.accept?.includes("application/json")) {
      return res.status(500).json({ success: false });
    }
    req.session.toast = {
      type: "error",
      message: "Failed to rename video.",
    };
    res.redirect("/history");
  }
};
