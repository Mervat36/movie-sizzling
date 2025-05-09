const mongoose = require("mongoose");

const UserQuerySchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: false }, // optional
  query: { type: String, required: true },   // text input from user
  createdAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model("UserQuery", UserQuerySchema);
