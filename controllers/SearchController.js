// controllers/searchController.js
const fs = require("fs");
const path = require("path");
const { exec, execSync } = require("child_process");
const Video = require("../models/Video");
const UserQuery = require("../models/UserQuery");
const ResultVideo = require("../models/ResultVideo");
const { searchEngine, saveSearchHistory } = require("../utils/searchEngine");

exports.searchUser = async (req, res) => {
  const { query } = req.body;
  const userId = req.user?._id || req.session.user?._id;
  const videoId = req.body.videoId;

  // ✅ Get videoTitle from session
  const safeTitle = (req.session.videoTitle || "").trim().replace(/ /g, "_");

  if (!safeTitle) {
    return res.status(400).send("No video linked to this search.");
  }

  try {
    const newQuery = new UserQuery({ userId, query, videoId });

    await newQuery.save();

    const captionsPath = path.join(
      __dirname,
      "..",
      `${safeTitle}_captions.json`
    );
    const videoPath = path.join("uploads", `dl_${safeTitle}.mp4`);
    console.log("📁 Looking for captions at:", captionsPath);
    console.log("📁 Looking for video at:", videoPath);

    if (!fs.existsSync(captionsPath) || !fs.existsSync(videoPath)) {
      return res.status(400).render("error", {
        error: { status: 400, message: "Missing captions or video." },
        theme: req.session.theme || "light",
        friendlyMessage:
          "The video or its subtitles couldn't be found. Please re-upload your video.",
      });
    }

    const pyCommand = `set PYTHONPATH=. && venv_search\\Scripts\\python.exe AI/search/search_user_query.py "${captionsPath}" "${query}" "${videoPath}"`;

    exec(pyCommand, async (error, stdout, stderr) => {
      if (error) {
        console.error("❌ Python search failed:", error.message);
        console.error("🔴 STDERR:", stderr);
        return res.status(500).render("error", {
          error: { status: 500, message: "Search processing failed." },
          theme: req.session.theme || "light",
          friendlyMessage:
            "There was a problem analyzing your search. Please try again later.",
        });
      }

      console.log("✅ Search finished:", stdout);

      let parsedResults;
      try {
        parsedResults = JSON.parse(stdout);
        console.log("📦 Parsed Results Count:", parsedResults.length);
        console.log("📝 First result:", parsedResults[0]);
      } catch (e) {
        console.error("❌ Failed to parse Python output:", e.message);
        return res.status(500).render("error", {
          error: { status: 500, message: "Search result parse error" },
          theme: req.session.theme || "light",
          friendlyMessage:
            "We couldn't understand the search results. Please try again.",
        });
      }

      req.session.searchResults = parsedResults;
      req.session.searchQuery = query;
      res.redirect(307, "/api/search/result");
    });
  } catch (err) {
    console.error("❌ Error saving query or executing search:", err.message);
    return res.status(500).render("error", {
      error: { status: 500, message: "Unexpected server error." },
      theme: req.session.theme || "light",
      friendlyMessage:
        "Something went wrong on our end. Please try again shortly.",
    });
  }
};

exports.searchResult = async (req, res) => {
  const userQuery = req.session.searchQuery || "";
  let videoTitle;
  if (req.body.videoId) {
    const video = await Video.findById(req.body.videoId);
    videoTitle = video?.filename.replace(/^dl_/, "").replace(/\.mp4$/, "");
  } else {
    videoTitle = req.session.videoTitle || req.body.videoTitle || null;
  }

  req.session.videoTitle = videoTitle;

  if (!videoTitle || !userQuery) {
    return res.status(400).render("error", {
      error: {
        status: 400,
        message: "Missing video or search input",
      },
      theme: req.session.theme || "light",
      friendlyMessage:
        "Oops! We couldn’t find your video or search term. Please try uploading a video again or go back to the search page.",
    });
  }

  const captionPath = path.join(__dirname, "..", `${videoTitle}_captions.json`);

  if (!fs.existsSync(captionPath)) {
    return res.status(404).send("Caption data not found.");
  }

  try {
    console.log("🔍 SESSION searchResults:", req.session.searchResults?.length);
    console.log("🔍 SESSION searchQuery:", req.session.searchQuery);
    let allResults = req.session.searchResults || [];
    allResults = allResults
      .filter((result) => result.score >= 0.4)
      .sort((a, b) => b.score - a.score);

    const perPage = 5;
    const currentPage = parseInt(req.query.page) || 1;
    const userQuery = req.session.searchQuery || "";

    const { exec, execSync } = require("child_process");

    const inputPath = path.resolve("uploads", `dl_${videoTitle}.mp4`);
    const clipsDir = path.join("output", "clips");
    if (!fs.existsSync(clipsDir)) {
      fs.mkdirSync(clipsDir, { recursive: true });
    }

    const trimmedResults = [];

    for (const match of allResults) {
      const clipName = `${videoTitle}_${match.start_time.replace(
        /:/g,
        "-"
      )}_${match.end_time.replace(/:/g, "-")}_clip.mp4`;
      const outputClipPath = path.resolve(clipsDir, clipName);
      const clipUrl = `/output/clips/${clipName.replace(/\\/g, "/")}`;

      if (!fs.existsSync(outputClipPath)) {
        const trimCommand = `ffmpeg -y -i "${inputPath}" -ss "${match.start_time}" -to "${match.end_time}" -vcodec libx264 -acodec aac -strict experimental -preset ultrafast -crf 28 "${outputClipPath}"`;

        try {
          execSync(trimCommand);
          if (
            !fs.existsSync(outputClipPath) ||
            fs.statSync(outputClipPath).size === 0
          )
            continue;
        } catch (error) {
          console.error("❌ FFmpeg failed:", error.message);
          continue;
        }
      }

      trimmedResults.push({ ...match, clip: clipUrl });
    }
    const start = (currentPage - 1) * perPage;
    const paginatedResults = trimmedResults.slice(start, start + perPage);
    const totalPages = Math.ceil(trimmedResults.length / perPage);
    if (paginatedResults.length === 0) {
      return res.render("results", {
        results: [],
        query: userQuery,
        video: `/uploads/dl_${videoTitle}.mp4`,
        message: "No results matched your query.",
      });
    }
    console.log("📥 Using results from session:", paginatedResults.length);
    req.session.trimmedResults = trimmedResults;
    req.session.paginatedIndex = paginatedResults.length;
    req.session.initialResultsSaved = true; // now we are saving the first 5 right here

    // ✅ Save first 5 shown results into history
    if (paginatedResults.length > 0 && req.session.user) {
      const userId = req.session.user._id;
      const video = await Video.findOne({ filename: `dl_${videoTitle}.mp4` });

      if (video) {
        let query = await UserQuery.findOne({
          userId,
          videoId: video._id,
          query: userQuery,
        });

        if (!query) {
          query = await UserQuery.create({
            userId,
            videoId: video._id,
            query: userQuery,
          });
        }

        for (const match of paginatedResults) {
          const clipFilename = match.clip.split("/").pop();
          const alreadySaved = await ResultVideo.findOne({
            queryId: query._id,
            clipFilename,
            timeRange: `${match.start_time} - ${match.end_time}`,
          });

          if (!alreadySaved) {
            await ResultVideo.create({
              queryId: query._id,
              clipFilename,
              timeRange: `${match.start_time} - ${match.end_time}`,
              caption: match.caption,
            });
          }
        }
      }
    }

    res.render("results", {
      results: paginatedResults,
      query: userQuery,
      video: null,
      message: null,
      currentPage,
      totalPages,
    });
  } catch (err) {
    console.error("❌ Failed to process search results:", err.message);
    return res.status(500).send("Server error processing results.");
  }
};

exports.renderSearchPage = async (req, res) => {
  const videoId = req.query.videoId;

  if (!videoId) {
    return res.render("search", {
      videoId: null,
      videoFilename: null,
      videoTitle: null,
    });
  }

  try {
    const video = await Video.findById(videoId);
    if (!video) {
      console.error("❌ Video not found for ID:", videoId);
      return res.render("search", {
        videoId,
        videoFilename: null,
        videoTitle: null,
      });
    }

    return res.render("search", {
      videoId,
      videoFilename: video.filename,
      videoTitle: video.title || video.filename,
    });
  } catch (err) {
    console.error("❌ Error fetching video:", err.message);
    return res.render("search", {
      videoId,
      videoFilename: null,
      videoTitle: null,
    });
  }
};

exports.searchSubmit = async (req, res) => {
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

    const rawTitle = video.title || video.filename;
    const safeTitle = rawTitle.trim().replace(/ /g, "_");

    req.session.videoTitle = safeTitle;

    const result = await searchEngine(query, videoId);
    const userId = req.session.user?._id || "guest";

    await saveSearchHistory(userId, videoId, query, result);

    res.redirect(`/search?videoId=${videoId}`);
  } catch (error) {
    console.error("Search error:", error);
    res.status(500).send("Search error");
  }
};

exports.rerunSearch = async (req, res) => {
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
};
exports.showMoreResults = async (req, res) => {
  try {
    const allResults = req.session.trimmedResults || [];
    let currentIndex = req.session.paginatedIndex || 0;
    const perPage = 5;

    const nextBatch = allResults.slice(currentIndex, currentIndex + perPage);
    req.session.paginatedIndex = currentIndex + nextBatch.length;

    const done = req.session.paginatedIndex >= allResults.length;

    res.json({ results: nextBatch, done });

    // 🔁 Save initial + next batch on first click
    if (req.session.user && nextBatch.length > 0) {
      const userId = req.session.user._id;
      const videoTitle = req.session.videoTitle;
      const video = await Video.findOne({ filename: `dl_${videoTitle}.mp4` });

      if (!video) return;

      let query = await UserQuery.findOne({
        userId,
        videoId: video._id,
        query: req.session.searchQuery,
      });

      if (!query) {
        query = await UserQuery.create({
          userId,
          videoId: video._id,
          query: req.session.searchQuery,
        });
      }

      // 🔁 Combine initial 5 + current batch if not saved yet
      const resultsToSave = req.session.initialResultsSaved
        ? nextBatch
        : allResults.slice(0, currentIndex + perPage);

      for (const match of resultsToSave) {
        const clipFilename = match.clip.split("/").pop();
        const alreadySaved = await ResultVideo.findOne({
          queryId: query._id,
          clipFilename,
          timeRange: `${match.start_time} - ${match.end_time}`,
        });

        if (!alreadySaved) {
          await ResultVideo.create({
            queryId: query._id,
            clipFilename,
            timeRange: `${match.start_time} - ${match.end_time}`,
            caption: match.caption,
          });
        }
      }

      req.session.initialResultsSaved = true; // mark first 5 as saved
    }
  } catch (err) {
    console.error("❌ Error in showMoreResults:", err.message);
    res.status(500).json({ error: "Failed to load more results." });
  }
};
