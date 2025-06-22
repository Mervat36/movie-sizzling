const mongoose = require("mongoose");

const ReportSchema = new mongoose.Schema({
  video: { type: mongoose.Schema.Types.ObjectId, ref: "Video", required: true },
  reportedBy: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
  reason: { type: String, required: true },
  status: { type: String, enum: ["pending", "resolved"], default: "pending" }, // New field
}, { timestamps: true });

module.exports = mongoose.model("Report", ReportSchema);
