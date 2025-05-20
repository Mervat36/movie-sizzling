// 📁 routes/HistoryRoutes.js
const express = require("express");
const router = express.Router();
const {
  getHistoryPage,
  downloadVideo,
  deleteVideo,
  deleteQuery,
  downloadResult,
  deleteResult,
} = require("../controllers/HistoryController");

const { ensureAuthenticated } = require("../middleware/auth");

// Routes
router.get("/history", ensureAuthenticated, getHistoryPage);
router.get("/videos/download/:id", ensureAuthenticated, downloadVideo);
router.post("/videos/delete/:id", ensureAuthenticated, deleteVideo);
router.post("/queries/delete/:id", ensureAuthenticated, deleteQuery);
router.get("/results/download/:id", ensureAuthenticated, downloadResult);
router.post("/results/delete/:id", ensureAuthenticated, deleteResult);

module.exports = router;
