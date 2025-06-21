const express = require('express');
const router = express.Router();
const AdminController = require('../controllers/AdminController');
const { isAdmin } = require('../middleware/auth');

router.get('/admin', isAdmin, AdminController.renderAdminPage);

module.exports = router; 