const Search = require("../models/Search");

// 1. Return the user's past search history.
exports.getUserHistory = async (req, res) => {
  try {
    const page = parseInt(req.query.page) || 1;
    const limit = 2;
    const skip = (page - 1) * limit;

    const totalCount = await Search.countDocuments({ userId: req.user.id });
    const totalPages = Math.ceil(totalCount / limit);

    const history = await Search.find({ userId: req.user.id })
      .sort({ createdAt: -1 })
      .skip(skip)
      .limit(limit);

    if (!history.length) {
      return res.status(404).json({ message: "No history found" });
    }

    res.status(200).json({
      currentPage: page,
      totalPages,
      totalCount,
      perPage: limit,
      results: history
    });
  } catch (error) {
    res.status(500).json({ message: "Error fetching history", error });
  }
};

// 2. Clear User Search History.
exports.clearHistory = async (req, res) => {
  try {
    await Search.deleteMany({ userId: req.user.id });
    res.status(200).json({ message: "Search history cleared successfully!" });
  } catch (error) {
    res.status(500).json({ message: "Error clearing history", error });
  }
};
