const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

const ShotMetadata = require("../models/ShotData");
const Video = require("../models/Video");
const SceneMetadata = require("../models/SceneMetadata");
const SceneResults = require("../models/SceneSearchResult");
const ResultVideo = require("../models/ResultVideo");
const SceneMatchResult = require("../models/SceneMatchResult");

// === 1. Upload and Process Video ===
exports.uploadVideo = async (req, res) => {
  try {
    const videoPath = req.file.path;
    const movieTitle = req.body.title || path.parse(req.file.originalname).name;

    const scriptPath = path.join(
      __dirname,
      "../AI/shot_segmentation/shot_segmentation.py"
    );
    const python = spawn("python", [scriptPath, videoPath, movieTitle]);

    python.stderr.on("data", (err) => console.error(`PYTHON ERROR: ${err}`));

    python.on("close", async () => {
      const jsonPath = path.join("output", `${movieTitle}_shots.json`);
      if (!fs.existsSync(jsonPath)) {
        return res
          .status(500)
          .json({ message: "Shot metadata not generated." });
      }

      const shotData = JSON.parse(fs.readFileSync(jsonPath, "utf-8"));
      try {
        const saved = await ShotMetadata.create(shotData);

        // Scene Segmentation
        const sceneScript = path.join(
          __dirname,
          "../AI/Scene/scene_segmentation.py"
        );
        const sceneInput = path.join("output", movieTitle);
        const modelPath = path.join("AI", "Scene", "scene_model.pth");
        const sceneOutput = path.join("output", `${movieTitle}_scenes.json`);

        const sceneProcess = spawn("python", [
          sceneScript,
          sceneInput,
          modelPath,
          sceneOutput,
        ]);
        sceneProcess.stderr.on("data", (err) =>
          console.error(`SCENE ERROR: ${err}`)
        );
        sceneProcess.on("close", () =>
          res.redirect(`/search?videoId=${saved.movieTitle || saved._id}`)
        );
      } catch (err) {
        console.error("MongoDB save error:", err);
        return res
          .status(500)
          .json({ message: "Failed to save metadata", error: err });
      }
    });
  } catch (error) {
    console.error("Upload error:", error);
    res.status(500).json({ message: "Error uploading video", error });
  }
};

// === 2. Get All Videos ===
exports.getAllVideos = async (req, res) => {
  try {
    const videos = await Video.find();
    res.status(200).json(videos);
  } catch (error) {
    res.status(500).json({ message: "Error fetching videos", error });
  }
};

// === 3. Get Video By ID ===
exports.getVideoById = async (req, res) => {
  try {
    const video = await Video.findById(req.params.id);
    if (!video) return res.status(404).json({ message: "Video not found" });
    res.status(200).json(video);
  } catch (error) {
    res.status(500).json({ message: "Error fetching video", error });
  }
};

// === 4. Delete Video + All Related Data ===
exports.deleteVideoByTitle = async (req, res) => {
  const { title } = req.params;
  try {
    const deletedVideo = await Video.findOneAndDelete({ title });
    if (!deletedVideo)
      return res.status(404).json({ message: "Video not found" });

    await ShotMetadata.deleteMany({ movieTitle: title });
    await SceneMetadata.deleteMany({ title });
    await SceneResults.deleteMany({ movie_name: title });

    const resultVideos = await ResultVideo.find({}).populate("queryId");
    const clipFiles = [];

    for (const result of resultVideos) {
      if (result.queryId?.result?.title === title) {
        clipFiles.push(result.clipFilename);
        await ResultVideo.findByIdAndDelete(result._id);
        await SceneMatchResult.findByIdAndDelete(result.queryId._id);
      }
    }

    // Delete result clips
    clipFiles.forEach((file) => {
      const filePath = path.join(__dirname, "../uploads/results", file);
      if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
    });

    // Delete main video file
    const videoPath = path.join(__dirname, "../uploads", `${title}.mp4`);
    if (fs.existsSync(videoPath)) fs.unlinkSync(videoPath);

    // Delete AI output
    [
      path.join("output", `${title}_shots.json`),
      path.join("output", `${title}_scenes.json`),
    ].forEach((file) => {
      if (fs.existsSync(file)) fs.unlinkSync(file);
    });
    const shotFolder = path.join("output", title);
    if (fs.existsSync(shotFolder))
      fs.rmSync(shotFolder, { recursive: true, force: true });
    // Delete optional scene summary file(s)
    const sceneSummaryVariants = [
      `${title}_scene_summaries.json`,
      `${safeTitle}_scene_summaries.json`,
      `${title}_summary.json`,
    ];

    sceneSummaryVariants.forEach((file) => {
      const filePath = path.join(__dirname, "..", file);
      if (fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
      }
    });

    res.json({
      success: true,
      message: `Deleted video "${title}" and all related data.`,
    });
  } catch (error) {
    console.error("❌ Deletion error:", error);
    res.status(500).json({ success: false, error: error.message });
  }
};
