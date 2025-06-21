const mongoose = require("mongoose");

const userQuerySchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
  videoId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "Video",
    required: true,
  },
  query: { type: String, required: true },
  timestamp: { type: Date, default: Date.now },
  customTitle: { type: String, default: null }
});

module.exports = mongoose.model("UserQuery", userQuerySchema);
