const path = require("path");
const { exec } = require("child_process");
const UserQuery = require("../models/UserQuery");
const ResultVideo = require("../models/ResultVideo"); // ✅ NEW: import your new model

exports.showResult = async (req, res) => {
  try {
    const { videoId, segment, q } = req.query;
    const { start, end } = JSON.parse(segment);

    const inputPath = path.join(__dirname, "../uploads", `${videoId}.mp4`);
    const outputFile = `${videoId}_${start.replace(/:/g, "-")}_${end.replace(
      /:/g,
      "-"
    )}.mp4`;
    const outputPath = path.join(
      __dirname,
      "../public/output/clips",
      outputFile
    );
    const videoUrl = `/output/clips/${outputFile}`;

    const command = `ffmpeg -i "${inputPath}" -ss ${start} -to ${end} -c copy "${outputPath}"`;

    exec(command, async (err) => {
      if (err) {
        console.error("FFmpeg error:", err);
        return res.status(500).send("Error generating video clip");
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
