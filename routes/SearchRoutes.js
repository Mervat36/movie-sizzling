const express = require('express');
const router = express.Router();
const UserQuery = require('../models/UserQuery');

// Route to handle POST request from form
router.post('/', async (req, res) => {
  const { query } = req.body;

  try {
    // Save the query to MongoDB
    const newQuery = new UserQuery({ query });
    await newQuery.save();

    // Optionally, fetch results here or redirect
    res.render('searchResults', { query });
  } catch (error) {
    console.error(error);
    res.status(500).send('Server error');
  }
});

module.exports = router;
