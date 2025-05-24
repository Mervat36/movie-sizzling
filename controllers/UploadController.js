const multer = require("multer");
const Video = require("../models/Video");

// 1. Configures storage path and file naming for uploaded videos.
const storage = multer.diskStorage({
  destination: "./public/uploads/",
  filename: (req, file, cb) => {
    const rawTitle = req.body.title || "untitled";
    const safeTitle = rawTitle.trim().replace(/[^a-z0-9_\-]/gi, "_").toLowerCase();
    const finalName = `dl_${safeTitle}.mp4`;
    req.body.safeFilename = finalName; // Pass to controller
    req.body.safeTitle = safeTitle;
    cb(null, finalName);
  },
});

const upload = multer({ storage }).single("video");

// 3. Uploads a video and saves its metadata into the database.
exports.uploadVideo = async (req, res) => {
  try {
    const file = req.file;
    const rawTitle = req.body.title || file.originalname;
    const safeFilename = req.body.safeFilename;
    const safeTitle = req.body.safeTitle;

    if (!file) return res.status(400).send("No file uploaded");

    const newVideo = new Video({
      filename: safeFilename,
      originalName: file.originalname,
      title: rawTitle,
      user: req.user?._id,
      createdAt: new Date(),
    });

    await newVideo.save();

    // Optional: Store safe title in session for later caption logic
    req.session.videoTitle = safeTitle;

    res.redirect("/history");
  } catch (error) {
    console.error("Upload error:", error);
    res.status(500).send("Server Error");
  }
};
