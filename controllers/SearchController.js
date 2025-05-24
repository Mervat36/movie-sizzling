// controllers/searchController.js
const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");
const Video = require("../models/Video");
const UserQuery = require("../models/UserQuery");
const ResultVideo = require("../models/ResultVideo");
const { searchEngine, saveSearchHistory } = require("../utils/searchEngine");

exports.searchUser = async (req, res) => {
  const { query } = req.body;
  const userId = req.user?._id || req.session.user?._id;

  // ✅ Get videoTitle from session
  const safeTitle = req.session.videoTitle;

  if (!safeTitle) {
    return res.status(400).send("No video linked to this search.");
  }

  try {
    const newQuery = new UserQuery({ userId, query });
    await newQuery.save();

    const captionsPath = path.join(__dirname, `${safeTitle}_captions.json`);
    const videoPath = path.join("uploads", `dl_${safeTitle}.mp4`);

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
      res.redirect("/search-results");
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
  const userQuery = req.body.query;
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
    const rawData = fs.readFileSync(captionPath, "utf-8").trim();

    if (rawData.startsWith("<")) {
      console.error("❌ Captions file contains HTML, not JSON");
      return res.status(500).render("error", {
        error: { status: 500, message: "Corrupted captions file" },
        theme: req.session.theme || "light",
        friendlyMessage:
          "Something went wrong while reading the video data. Please try re-uploading your video.",
      });
    }

    let parsed;
    try {
      parsed = JSON.parse(rawData);
    } catch (e) {
      console.error("❌ Failed to parse captions JSON:", e.message);
      return res.status(500).render("error", {
        error: { status: 500, message: "Invalid JSON format" },
        theme: req.session.theme || "light",
        friendlyMessage:
          "We couldn’t process your video results. Please try again later.",
      });
    }

    const metadata = parsed.shots_metadata;

    const results = [];

    for (const [imgPath, shot] of Object.entries(metadata)) {
      if (shot.caption.toLowerCase().includes(userQuery.toLowerCase())) {
        results.push({
          caption: shot.caption,
          tags: shot.tags,
          start_time: shot.start_time,
          end_time: shot.end_time,
          image: imgPath,
        });
      }
    }

    if (results.length === 0) {
      return res.render("results", {
        results: [],
        query: userQuery,
        video: `/uploads/dl_${videoTitle}.mp4`,
        message: "No results matched your query.",
      });
    }

    const { execSync } = require("child_process");

    const inputPath = path.join("uploads", `dl_${videoTitle}.mp4`);
    const clipsDir = path.join("output", "clips");
    if (!fs.existsSync(clipsDir)) {
      fs.mkdirSync(clipsDir, { recursive: true });
    }

    const trimmedResults = [];

    for (const match of results) {
      const clipName = `${videoTitle}_${match.start_time.replace(
        /:/g,
        "-"
      )}_${match.end_time.replace(/:/g, "-")}_clip.mp4`;
      const outputClipPath = path.join(clipsDir, clipName);
      const clipUrl = `/output/clips/${clipName}`;

      if (!fs.existsSync(outputClipPath)) {
        const trimCommand = `ffmpeg -y -i "${inputPath}" -ss ${match.start_time} -to ${match.end_time} -preset ultrafast -crf 28 "${outputClipPath}"`;
        try {
          execSync(trimCommand);
          console.log("✅ Created clip:", clipName);
        } catch (error) {
          console.error("❌ Failed to trim clip:", error.message);
          continue; // Skip this one if failed
        }
      }

      trimmedResults.push({
        ...match,
        clip: clipUrl,
      });
    }

    const clipToSave = trimmedResults[0]?.clip || null;

    if (trimmedResults.length > 0 && req.session.user) {
      try {
        // Save the user query
        const newQuery = await UserQuery.create({
          userId: req.session.user._id,
          videoId: req.body.videoId,
          query: userQuery,
        });

        // Save each result video
        for (const match of trimmedResults) {
          const clipFilename = match.clip.split("/").pop();
          await ResultVideo.create({
            queryId: newQuery._id,
            clipFilename,
            timeRange: `${match.start_time} - ${match.end_time}`,
            caption: match.caption,
          });
        }

        console.log("✅ Query and results saved to history.");
      } catch (err) {
        console.warn("⚠️ Failed to save query/results:", err.message);
      }
    }
    res.render("results", {
      results: trimmedResults,
      query: userQuery,
      video: null,
      message: null,
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

    const safeTitle = video.filename.replace(/^dl_/, "").replace(/\.mp4$/, "");
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
