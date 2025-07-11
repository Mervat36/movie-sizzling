// controllers/searchController.js
const fs = require("fs");
const path = require("path");
const { exec, execSync } = require("child_process");
const Video = require("../models/Video");
const UserQuery = require("../models/UserQuery");
const ResultVideo = require("../models/ResultVideo");
const { searchEngine, saveSearchHistory } = require("../utils/searchEngine");
const Caption = require("../models/Caption");

const timestampToSeconds = (ts) => {
  if (!ts) return 0;
  const parts = ts.split(':').map(Number);
  if (parts.length === 3) {
    return parts[0] * 3600 + parts[1] * 60 + parts[2];
  }
  return 0;
};

exports.searchUser = async (req, res) => {
  // Reset previous session search data
  req.session.searchResults = null;
  req.session.trimmedResults = null;
  req.session.paginatedIndex = 0;
  req.session.initialResultsSaved = false;
  req.session.currentQueryId = null;
  const { query } = req.body;
  const userId = req.user?._id || req.session.user?._id;
  const videoId = req.body.videoId;
  // ✅ Get videoTitle from session
  const safeTitle = (req.session.videoTitle || "").trim().replace(/ /g, "_");

  if (!safeTitle) {
    return res.status(400).send("No video linked to this search.");
  }

  try {
    const newQuery = await UserQuery.create({ userId, query, videoId });
    req.session.currentQueryId = newQuery._id;

    const captionsPath = path.join(__dirname, "..", `${safeTitle}_captions.json`);
    const videoPath = path.join("uploads", `dl_${safeTitle}.mp4`);

    // 🔁 Try to download missing captions from MongoDB (using Caption model)
    if (!fs.existsSync(captionsPath)) {
      const captionDoc = await Caption.findOne({ movie_name: safeTitle });
      if (captionDoc) {
        fs.writeFileSync(captionsPath, JSON.stringify({
          movie_name: captionDoc.movie_name,
          shots_metadata: captionDoc.shots_metadata
        }, null, 2));
        console.log("✅ Downloaded caption JSON from MongoDB.");
      } else {
        console.error("❌ Captions not found in DB.");
        return res.status(400).render("error", {
          error: { status: 400, message: "Captions not found." },
          theme: req.session.theme || "light",
          friendlyMessage: "This video doesn't have subtitle data yet.",
        });
      }
    }

    // 🧱 Video download is already handled earlier – just validate here
    if (!fs.existsSync(videoPath)) {
      return res.status(400).render("error", {
        error: { status: 400, message: "Video not found." },
        theme: req.session.theme || "light",
        friendlyMessage: "Video file is missing. Try reloading from catalog.",
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
        if (stdout.trim().startsWith("<html")) {
          throw new Error(
            "Received HTML instead of JSON. Possible server misconfig."
          );
        }
        parsedResults = JSON.parse(stdout);
        console.log("📦 Parsed Results Count:", parsedResults.length);
        if (!Array.isArray(parsedResults) || parsedResults.length === 0) {
          return res.status(200).render("results", {
            results: [],
            query,
            video: `/uploads/dl_${safeTitle}.mp4`,
            message: "No relatable scenes were found.",
          });
        }
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
        "Oops! We couldn't find your video or search term. Please try uploading a video again or go back to the search page.",
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

    const inputPath = path.resolve("uploads", `dl_${videoTitle}.mp4`);
    const clipsDir = path.join("output", "clips");
    if (!fs.existsSync(clipsDir)) {
      fs.mkdirSync(clipsDir, { recursive: true });
    }

    // Sort by start time to check for sequential scenes
    allResults.sort((a, b) => timestampToSeconds(a.start_time) - timestampToSeconds(b.start_time));

    const sequentialGroups = [];
    let currentGroup = [];

    for (let i = 0; i < allResults.length; i++) {
      if (currentGroup.length === 0) {
        currentGroup.push(allResults[i]);
      } else {
        const lastInGroup = currentGroup[currentGroup.length - 1];
        const currentTime = allResults[i];
        
        // Check if the current scene starts immediately after the last one (with a small tolerance)
        const lastEndTime = timestampToSeconds(lastInGroup.end_time);
        const currentStartTime = timestampToSeconds(currentTime.start_time);

        if (Math.abs(currentStartTime - lastEndTime) < 2) { // 2-second tolerance
          currentGroup.push(currentTime);
        } else {
          if (currentGroup.length > 1) {
            sequentialGroups.push(currentGroup);
          }
          currentGroup = [allResults[i]];
        }
      }
    }
    if (currentGroup.length > 1) {
      sequentialGroups.push(currentGroup);
    }
    
    // Process concatenated clips
    const concatenatedClips = [];
    for (const group of sequentialGroups) {
      const firstScene = group[0];
      const lastScene = group[group.length - 1];
      const concatClipName = `${videoTitle}_${firstScene.start_time.replace(/:/g, "-")}_to_${lastScene.end_time.replace(/:/g, "-")}_concat.mp4`;
      const concatClipPath = path.resolve(clipsDir, concatClipName);
      const concatClipUrl = `/output/clips/${concatClipName}`;
      
      if (!fs.existsSync(concatClipPath)) {
        const fileListPath = path.join(clipsDir, `concat_list_${Date.now()}.txt`);
        let fileListContent = "";
        group.forEach(scene => {
            const clipName = `${videoTitle}_${scene.start_time.replace(/:/g, "-")}_${scene.end_time.replace(/:/g, "-")}_clip.mp4`;
            const clipPath = path.resolve(clipsDir, clipName);
            if(fs.existsSync(clipPath)){
                fileListContent += `file '${clipPath}'\n`;
            }
        });

        fs.writeFileSync(fileListPath, fileListContent);

        const concatCommand = `ffmpeg -f concat -safe 0 -i "${fileListPath}" -c copy "${concatClipPath}"`;
        try {
          execSync(concatCommand);
          fs.unlinkSync(fileListPath); // Clean up the temporary file list
        } catch (error) {
          console.error("❌ FFmpeg concatenation failed:", error.message);
          continue; // Skip if concatenation fails
        }
      }

      concatenatedClips.push({
        clip: concatClipUrl,
        caption: `Combined scene: ${group.map(s => s.caption).join(' ... ')}`,
        start_time: firstScene.start_time,
        end_time: lastScene.end_time,
        score: group.reduce((acc, s) => acc + s.score, 0) / group.length,
      });
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
        const queryId = req.session.currentQueryId;
        if (queryId) {
          const query = await UserQuery.findById(queryId);
          if (query) {
            for (const match of paginatedResults) {
              const clipFilename = match.clip.split("/").pop();
              const alreadySaved = await ResultVideo.findOne({
                queryId,
                clipFilename,
                timeRange: `${match.start_time} - ${match.end_time}`,
              });

              if (!alreadySaved) {
                await ResultVideo.create({
                  queryId,
                  clipFilename,
                  timeRange: `${match.start_time} - ${match.end_time}`,
                  caption: match.caption,
                });
              }
            }
          }
        }
      }
    }
    const queryId = req.session.currentQueryId || null;
    const done = paginatedResults.length + start >= trimmedResults.length;

    res.render("results", {
      concatenatedClips,
      results: paginatedResults,
      query: userQuery,
      video: null,
      message: null,
      currentPage,
      totalPages,
      queryId,
      done,
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

      const queryId = req.session.currentQueryId;
      const query = queryId ? await UserQuery.findById(queryId) : null;
      if (!query) return;

      // 🔁 Combine initial 5 + current batch if not saved yet
      const resultsToSave = req.session.initialResultsSaved
        ? nextBatch
        : allResults.slice(0, currentIndex + perPage);

      for (const match of resultsToSave) {
        const clipFilename = match.clip.split("/").pop();
        const alreadySaved = await ResultVideo.findOne({
          queryId: queryId,
          clipFilename,
          timeRange: `${match.start_time} - ${match.end_time}`,
        });

        if (!alreadySaved) {
          await ResultVideo.create({
            queryId: queryId,
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
