const mongoose = require("mongoose");

const VideoSchema = new mongoose.Schema(
  {
    user: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
    title: { type: String, required: true },
    url: { type: String },
    description: { type: String, default: "" },
    filename: { type: String, required: true },
    originalName: { type: String },
    isHidden: { type: Boolean, default: false },
  },
  { timestamps: true }
);

module.exports = mongoose.model("Video", VideoSchema);
