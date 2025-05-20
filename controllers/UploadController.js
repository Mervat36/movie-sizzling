const multer = require("multer");
const Video = require("../models/Video");

// 1. Configures storage path and file naming for uploaded videos.
const storage = multer.diskStorage({
  destination: "./public/uploads/",
  filename: (req, file, cb) => {
    cb(null, Date.now() + "-" + file.originalname);
  },
});

// 2. Handles single video upload.
const upload = multer({ storage }).single("video");

// 3. Uploads a video and saves its metadata into the database.
exports.uploadVideo = async (req, res) => {
  try {
    const file = req.file;
    const title = req.body.title || file.originalname;

    if (!file) return res.status(400).send("No file uploaded");

    const newVideo = new Video({
      filename: file.filename,
      originalName: file.originalname,
      title: title,
      user: req.user?._id,
      createdAt: new Date(),
    });

    await newVideo.save(); // ✅ Save to DB

    res.redirect("/history"); // or success page
  } catch (error) {
    console.error("Upload error:", error);
    res.status(500).send("Server Error");
  }
};
