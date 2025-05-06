const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");
const ShotMetadata = require("../models/ShotData");
const Video = require("../models/Video");

// 1. Uploads a video and processes it with TransNetV2
exports.uploadVideo = async (req, res) => {
  try {
    const videoPath = req.file.path;
    const movieTitle = req.body.title || req.file.originalname.split(".")[0];

    // Run Python script with video path and title
    const scriptPath = path.join(__dirname, "../AI/shot_segmentation/shot_segmentation.py");
    const python = spawn("python", [scriptPath, videoPath, movieTitle]);


    python.stdout.on("data", (data) => {
      console.log(`PYTHON: ${data}`);
    });

    python.stderr.on("data", (err) => {
      console.error(`PYTHON ERROR: ${err}`);
    });

    python.on("close", async (code) => {
      const jsonPath = path.join("output", `${movieTitle}_shots.json`);

      if (!fs.existsSync(jsonPath)) {
        return res.status(500).json({ message: "Shot metadata not generated." });
      }

      // Parse and save metadata to MongoDB
      const shotData = JSON.parse(fs.readFileSync(jsonPath, "utf-8"));

      try {
        const saved = await ShotMetadata.create(shotData);
        return res.status(200).json({ message: "Video processed and saved", data: saved });
      } catch (err) {
        console.error("MongoDB save error:", err);
        return res.status(500).json({ message: "Failed to save metadata", error: err });
      }
    });
  } catch (error) {
    console.error("Upload error:", error);
    res.status(500).json({ message: "Error uploading video", error });
  }
};

// 2. Returns all videos (from Video collection)
exports.getAllVideos = async (req, res) => {
  try {
    const videos = await Video.find();
    res.status(200).json(videos);
  } catch (error) {
    res.status(500).json({ message: "Error fetching videos", error });
  }
};

// 3. Get single video by ID
exports.getVideoById = async (req, res) => {
  try {
    const video = await Video.findById(req.params.id);
    if (!video) {
      return res.status(404).json({ message: "Video not found" });
    }
    res.status(200).json(video);
  } catch (error) {
    res.status(500).json({ message: "Error fetching video", error });
  }
};
