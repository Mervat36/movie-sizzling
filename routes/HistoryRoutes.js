const express = require("express");
const router = express.Router();
const UserQuery = require("../models/UserQuery");
const { ensureAuthenticated } = require("../middleware/auth");

// GET /history - show user's search history (with pagination)
router.get("/history", ensureAuthenticated, async (req, res) => {
  try {
    const page = parseInt(req.query.page) || 1;
    const limit = 2;
    const skip = (page - 1) * limit;

    const totalCount = await UserQuery.countDocuments({ userId: req.user._id });
    const totalPages = Math.ceil(totalCount / limit);

    const queries = await UserQuery.find({ userId: req.user._id })
      .sort({ timestamp: -1 })
      .skip(skip)
      .limit(limit);

    res.render("history", {
      user: req.user,
      queries,
      currentPage: page,
      totalPages,
    });
  } catch (err) {
    console.error("❌ Failed to retrieve search history:", err);
    res.render("history", {
      user: req.user,
      queries: [],
      currentPage: 1,
      totalPages: 1
    });
  }
});

module.exports = router;
