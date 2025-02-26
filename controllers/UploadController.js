const multer = require("multer");
const Video = require("../models/Video");

// Configure Multer Storage
const storage = multer.diskStorage({
    destination: "./public/uploads/",
    filename: (req, file, cb) => {
        cb(null, Date.now() + "-" + file.originalname);
    }
});

const upload = multer({ storage }).single("video");

// Upload Video
exports.uploadVideo = (req, res) => {
    upload(req, res, async (err) => {
        if (err) {
            return res.status(500).json({ message: "Error uploading file", error: err });
        }

        try {
            const newVideo = new Video({
                title: req.body.title,
                url: `/uploads/${req.file.filename}`,
                description: req.body.description
            });

            await newVideo.save();
            res.status(201).json({ message: "Video uploaded successfully", video: newVideo });
        } catch (error) {
            res.status(500).json({ message: "Error saving video", error });
        }
    });
};
