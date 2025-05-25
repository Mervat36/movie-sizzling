const path = require("path");
const fs = require("fs");
const { exec } = require("child_process");
const UserQuery = require("../models/UserQuery");
const ResultVideo = require("../models/ResultVideo");
const Video = require("../models/Video");

exports.showResult = async (req, res) => {
  try {
    const { videoId, segment, q } = req.query;
    const { start, end } = JSON.parse(segment);

    const video = await Video.findById(videoId);
    if (!video) return res.status(404).send("Video not found");

    const inputPath = path.join(__dirname, "../uploads", video.filename);

    const baseName = video.filename.replace(/^dl_/, "").replace(/\.mp4$/, "");
    const outputFile = `${baseName}_${start.replace(/:/g, "-")}_${end.replace(
      /:/g,
      "-"
    )}_clip.mp4`;
    const outputDir = path.join(__dirname, "../output/clips");
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    const outputPath = path.join(outputDir, outputFile);

    const videoUrl = `/output/clips/${outputFile}`;

    const command = `ffmpeg -i "${inputPath}" -ss ${start} -to ${end} -c copy "${outputPath}"`;

    exec(command, async (err) => {
      if (err) {
        console.error("❌ FFmpeg execution failed:", err.message);
        return res.status(500).render("results", {
          videoPath: null,
          query: q || "Unknown Query",
          results: [],
          message:
            "We couldn't generate the video clip. Please try again. If the problem continues, contact support.",
        });
      }

      if (!fs.existsSync(outputPath)) {
        console.error("❌ Clip file not found after FFmpeg:", outputPath);
        return res.status(500).render("results", {
          videoPath: null,
          query: q || "Unknown Query",
          results: [],
          message:
            "The clip could not be found after processing. Please try again or re-upload the video.",
        });
      }
      try {
        if (req.session.user) {
          // ✅ 1. Save the query
          const newQuery = await UserQuery.create({
            userId: req.user._id,
            videoId: videoId,
            query: q || "Unknown Query",
          });

          // ✅ 2. Save the result linked to the query
          await ResultVideo.create({
            queryId: newQuery._id,
            clipFilename: outputFile,
            timeRange: `${start} - ${end}`,
          });
        }

        res.render("results", {
          videoPath: videoUrl,
          query: q || "Unknown Query",
          results: [],
        });
      } catch (saveErr) {
        console.error("Error saving to DB:", saveErr);
        res.render("results", {
          videoPath: videoUrl,
          query: q || "Unknown Query",
          results: [],
        });
      }
    });
  } catch (error) {
    console.error("Result generation error:", error);
    res.status(500).send("Server error");
  }
};
