const express = require("express");
const router = express.Router();
const { searchScenes, getSearchHistory } = require("../controllers/SearchController");

// Search Routes
router.get("/", searchScenes);
router.get("/history", getSearchHistory);

module.exports = router;
