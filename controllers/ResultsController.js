const path = require("path");
const { exec } = require("child_process");
const UserQuery = require("../models/UserQuery"); // ✅ Required for saving search history

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

    const command = `ffmpeg -i "${inputPath}" -ss ${start} -to ${end} -c copy "${outputPath}"`;

    exec(command, (err) => {
      if (err) {
        console.error("FFmpeg error:", err);
        return res.status(500).send("Error generating video clip");
      }

      const videoUrl = `/output/clips/${outputFile}`;

      // ✅ Save the search result in user history if logged in
      if (req.session.user) {
        UserQuery.create({
          userId: req.session.user._id,
          query: q || "Unknown Query",
          resultVideoUrl: videoUrl,
        })
          .then(() => {
            res.render("results", { videoPath: videoUrl });
          })
          .catch((saveErr) => {
            console.error("Error saving history:", saveErr);
            res.render("results", { videoPath: videoUrl });
          });
      } else {
        res.render("results", { videoPath: videoUrl });
      }
    });
  } catch (error) {
    console.error("Result generation error:", error);
    res.status(500).send("Server error");
  }
};
