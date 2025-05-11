// routes/history.js

const express = require("express");
const router = express.Router();
const UserQuery = require("../models/UserQuery");
const { ensureAuthenticated } = require("../middleware/auth");

// GET /history - show user's search history
router.get("/history", ensureAuthenticated, async (req, res) => {
  try {
    const queries = await UserQuery.find({ userId: req.user._id }).sort({ timestamp: -1 });
    res.render("history", { user: req.user, queries }); // Pass data to EJS
  } catch (err) {
    console.error("❌ Failed to retrieve search history:", err);
    res.render("history", { user: req.user, queries: [] }); // Show empty list on error
  }
});

module.exports = router;
