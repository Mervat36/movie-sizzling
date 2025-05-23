// 📁 controllers/historyController.js
const fs = require("fs");
const path = require("path");
const Video = require("../models/Video");
const ResultVideo = require("../models/ResultVideo");
const UserQuery = require("../models/UserQuery");

exports.getHistoryPage = async (req, res) => {
  const userId = req.user._id;
  const page = parseInt(req.query.page) || 1;
  const limit = 3; // 3 videos per page
  const skip = (page - 1) * limit;

  const totalVideos = await Video.countDocuments({ user: userId });
  const totalPages = Math.ceil(totalVideos / limit);

  const videos = await Video.find({ user: userId })
    .sort({ createdAt: -1 })
    .skip(skip)
    .limit(limit);

  const videoMap = {};
  for (const video of videos) {
    const queries = await UserQuery.find({ videoId: video._id });
    const queryMap = {};
    for (const query of queries) {
      const results = await ResultVideo.find({ queryId: query._id });
      queryMap[query._id] = { query, results };
    }
    videoMap[video._id] = { video, queries: queryMap };
  }

  const toast = req.session.toast || null;
  delete req.session.toast;

  res.render("history", {
    videoMap,
    toast,
    currentPage: page,
    totalPages,
  });
};

exports.downloadVideo = async (req, res) => {
  const video = await Video.findById(req.params.id);
  const filePath = path.join(__dirname, "../uploads", video.filename);
  res.download(filePath);
};

exports.deleteVideo = async (req, res) => {
  const videoId = req.params.id;
  const queries = await UserQuery.find({ videoId });

  for (const query of queries) {
    await ResultVideo.deleteMany({ queryId: query._id });
  }

  await UserQuery.deleteMany({ videoId });
  const video = await Video.findByIdAndDelete(videoId);

  const filePath = path.join(__dirname, "../uploads", video.filename);
  if (fs.existsSync(filePath)) fs.unlinkSync(filePath);

  if (
    req.headers.accept &&
    req.headers.accept.toLowerCase().includes("application/json")
  ) {
    res.setHeader("Content-Type", "application/json");
    return res.status(200).json({ success: true });
  }

  req.session.toast = {
    type: "success",
    message: "Video and related data deleted.",
  };
  res.redirect("/history");
};

exports.deleteQuery = async (req, res) => {
  const queryId = req.params.id;
  await ResultVideo.deleteMany({ queryId });
  await UserQuery.findByIdAndDelete(queryId);

  if (
    req.headers.accept &&
    req.headers.accept.toLowerCase().includes("application/json")
  ) {
    res.setHeader("Content-Type", "application/json");
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
  const result = await ResultVideo.findById(req.params.id);
  if (!result) {
    if (
      req.headers.accept &&
      req.headers.accept.toLowerCase().includes("application/json")
    ) {
      return res
        .status(404)
        .json({ success: false, message: "Result not found." });
    }

    req.session.toast = {
      type: "error",
      message: "Result not found or already deleted.",
    };
    return res.redirect("/history");
  }

  const clipFilename = result.clipFilename;

  // Delete all result entries that share this clip in the SAME query only
  await ResultVideo.deleteMany({
    queryId: result.queryId,
    clipFilename: clipFilename,
  });

  // Remove the clip file
  const clipPath = path.join(__dirname, "../public/output/clips", clipFilename);
  if (fs.existsSync(clipPath)) fs.unlinkSync(clipPath);

  if (
    req.headers.accept &&
    req.headers.accept.toLowerCase().includes("application/json")
  ) {
    res.setHeader("Content-Type", "application/json");
    return res.status(200).json({ success: true });
  }

  req.session.toast = {
    type: "success",
    message: "Result deleted from this query.",
  };
  res.redirect("/history");
};
