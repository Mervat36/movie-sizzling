const User = require('../models/User');
const Video = require('../models/Video');

exports.renderAdminPage = async (req, res) => {
  try {
    const users = await User.find().sort({ createdAt: -1 });
    const videos = await Video.find().sort({ createdAt: -1 });

    res.render('admin', {
      users,
      videos,
      layout: false, // We'll use a custom layout for the admin page
    });
  } catch (error) {
    console.error('Admin Page Error:', error);
    res.status(500).send('Server Error');
  }
}; 