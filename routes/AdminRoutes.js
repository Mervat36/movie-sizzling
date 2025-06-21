const express = require("express");
const router = express.Router();
const adminController = require("../controllers/AdminController");

// Admin dashboard page
router.get("/admin", adminController.getAdminDashboard);

// Video actions
router.post("/admin/delete-video/:id", adminController.deleteVideo);

// User actions
router.post("/admin/make-admin/:id", adminController.makeAdmin);
router.post("/admin/ban-user/:id", adminController.banUser);
router.post("/admin/delete-user/:id", adminController.deleteUser);
router.post("/admin/unban/:id", adminController.unbanUser);
router.post("/admin/make-admin/:id", adminController.makeAdmin);
router.post("/admin/remove-admin/:id", adminController.removeAdmin);

module.exports = router;
