const mongoose = require("mongoose");

const ShotMetaSchema = new mongoose.Schema({
  caption: { type: String, required: true },
  tags: {
    action: { type: String, default: null },
    place: { type: String, default: null },
    objects: { type: [String], default: [] },
  },
  start_time: { type: String, required: true },
  end_time: { type: String, required: true },
}, { _id: false });

const SceneSearchResultSchema = new mongoose.Schema({
  movie_name: { type: String, required: true },
  shots_metadata: {
    type: Object, // 👈 مهم جداً يكون كده
    required: true,
  },
});

module.exports = mongoose.models.SceneResults || mongoose.model("SceneResults", SceneSearchResultSchema);
