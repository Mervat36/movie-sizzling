const mongoose = require("mongoose");

const shotSchema = new mongoose.Schema({
  movieTitle: String,
  videoPath: String,
  fps: Number,
  shots: [
    {
      shotNumber: Number,
      startFrame: Number,
      middleFrame: Number,
      endFrame: Number,
      startTime: String,
      middleTime: String,
      endTime: String,
      images: {
        start: String,
        middle: String,
        end: String,
      },
    },
  ],
});

module.exports = mongoose.model("ShotMetadata", shotSchema);
