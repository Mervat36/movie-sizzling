const Search = require("../models/Search");

// 1. Searches for scenes in the database matching the query.
exports.searchScenes = async (req, res) => {
  try {
    const { query } = req.query;
    const results = await Search.find({
      sceneDescription: new RegExp(query, "i"),
    });

    if (!results.length) {
      return res.status(404).json({ message: "No matching scenes found" });
    }

    res.status(200).json(results);
  } catch (error) {
    res.status(500).json({ message: "Error searching scenes", error });
  }
};

// 2. Retrieves all search history.
exports.getSearchHistory = async (req, res) => {
  try {
    const history = await Search.find().sort({ createdAt: -1 });
    res.status(200).json(history);
  } catch (error) {
    res.status(500).json({ message: "Error fetching search history", error });
  }
};
