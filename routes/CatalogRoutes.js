const express = require("express");
const router = express.Router();
const { ensureAuthenticated } = require("../middleware/auth");
const CatalogController = require("../controllers/CatalogController");

router.post("/catalog/report/:videoId", ensureAuthenticated, CatalogController.reportVideo);
router.get("/catalog", ensureAuthenticated, CatalogController.renderCatalogPage);
router.get("/catalog/load/:id", ensureAuthenticated, CatalogController.handleCatalogClick);
router.post("/catalog/search", ensureAuthenticated, CatalogController.searchCatalog);

module.exports = router;
