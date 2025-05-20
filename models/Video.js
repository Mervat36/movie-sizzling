const mongoose = require("mongoose");

const VideoSchema = new mongoose.Schema(
  {
    user: { type: mongoose.Schema.Types.ObjectId, ref: "User" },
    title: String,
    url: String,
    description: String,
    filename: String
  },
  { timestamps: true }
);

module.exports = mongoose.model("Video", VideoSchema);
