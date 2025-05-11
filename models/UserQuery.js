// models/UserQuery.js
const mongoose = require("mongoose");

const userQuerySchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User" },
  query: String,
  resultVideoUrl: String,
  timestamp: { type: Date, default: Date.now },
});

module.exports = mongoose.model("UserQuery", userQuerySchema);