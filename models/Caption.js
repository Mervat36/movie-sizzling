const mongoose = require("mongoose");

const CaptionSchema = new mongoose.Schema({
  movie_name: String,
  shots_metadata: {
    type: mongoose.Schema.Types.Mixed,
    default: {}
  },
});

module.exports = mongoose.model("Caption", CaptionSchema); 