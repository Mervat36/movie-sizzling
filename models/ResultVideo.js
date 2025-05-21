// 📁 models/ResultVideo.js
const mongoose = require("mongoose");

const ResultVideoSchema = new mongoose.Schema({
  queryId: { type: mongoose.Schema.Types.ObjectId, ref: "UserQuery", required: true },
  clipFilename: { type: String, required: true },
  timeRange: { type: String },
  caption: { type: String },
  createdAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model("ResultVideo", ResultVideoSchema);
