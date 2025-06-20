const express = require("express");
const router = express.Router();
const { ensureAuthenticated } = require("../middleware/auth");
const { renderCatalogPage, handleCatalogClick, searchCatalog } = require("../controllers/CatalogController");

router.get("/catalog", ensureAuthenticated, renderCatalogPage);
router.get("/catalog/load/:id", ensureAuthenticated, handleCatalogClick);
router.post("/catalog/search", ensureAuthenticated, searchCatalog);

module.exports = router;
