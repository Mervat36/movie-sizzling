const mongoose = require("mongoose");

const VideoSchema = new mongoose.Schema(
  {
    title: String,
    url: String,
    description: String,
  },
  { timestamps: true }
);

module.exports = mongoose.model("Video", VideoSchema);
