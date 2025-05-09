const mongoose = require("mongoose");

const SceneSearchResultSchema = new mongoose.Schema({
  movie_name: String,
  shots_metadata: {
    type: Map,
    of: new mongoose.Schema({
      caption: String,
      tags: {
        action: String,
        place: String,
        objects: [String]
      },
      start_time: String,
      end_time: String
    }, { _id: false })
  }
});

module.exports = mongoose.model("SceneResults", SceneSearchResultSchema);
