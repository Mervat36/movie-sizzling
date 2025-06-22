const express = require("express");
const router = express.Router();
const adminController = require("../controllers/AdminController");
const { isAdmin } = require("../middleware/auth");

// Admin dashboard
router.get("/admin", isAdmin, adminController.getAdminDashboard);

// Video actions
router.post("/admin/delete-video/:id", adminController.deleteVideo);

// User actions
router.post("/admin/make-admin/:id", adminController.makeAdmin);
router.post("/admin/remove-admin/:id", adminController.removeAdmin);
router.post("/admin/ban-user/:id", adminController.banUser);
router.post("/admin/unban/:id", adminController.unbanUser);
router.post("/admin/delete-user/:id", adminController.deleteUser);
router.get("/admin/reports", isAdmin, adminController.getReports);
router.post("/admin/reports/delete/:reportId", isAdmin, adminController.deleteReport);
router.post("/admin/reports/resolve/:reportId", isAdmin, adminController.resolveReport);

module.exports = router;
