const mongoose = require("mongoose");

const SceneSchema = new mongoose.Schema({
  scene_id: String,
  start_shot: Number,
  end_shot: Number,
  start_time: String,
  end_time: String,
  thumbnail_path: String,
});

const SceneMetadataSchema = new mongoose.Schema({
  title: String,
  scenes: [SceneSchema],
});

module.exports = mongoose.model("SceneMetadata", SceneMetadataSchema);
