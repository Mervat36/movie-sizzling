const express = require("express");
const router = express.Router();
const { ensureAuthenticated } = require("../middleware/auth");
const searchController = require("../controllers/SearchController");

router.get("/", ensureAuthenticated, searchController.renderSearchPage);
router.get("/search-again/:videoId", ensureAuthenticated, searchController.rerunSearch);
router.post("/submit", ensureAuthenticated, searchController.searchSubmit);
router.post("/user", ensureAuthenticated, searchController.searchUser);
router.post("/result", ensureAuthenticated, searchController.searchResult);

module.exports = router;
