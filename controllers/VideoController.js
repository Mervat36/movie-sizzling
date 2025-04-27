const Video = require("../models/Video");

// 1. Uploads a video.
exports.uploadVideo = async (req, res) => {
    try {
        res.status(200).json({ message: "Video uploaded successfully!" });
    } catch (error) {
        res.status(500).json({ message: "Error uploading video", error });
    }
};

// Returns all videos.
exports.getAllVideos = async (req, res) => {
    try {
        const videos = await Video.find();
        res.status(200).json(videos);
    } catch (error) {
        res.status(500).json({ message: "Error fetching videos", error });
    }
};

// Get video by ID.
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
