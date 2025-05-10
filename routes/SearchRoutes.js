const express = require('express');
const router = express.Router();
const UserQuery = require('../models/UserQuery');
router.get('/result', require('../controllers/ResultsController').showResult);


// Route to handle POST request from form
const path = require('path');
const { exec } = require('child_process');
const { searchEngine, saveSearchHistory } = require('../utils/searchEngine'); // create this module

router.post('/result', async (req, res) => {
  const { query, videoId } = req.body;

  try {
    // Call your AI model or dummy logic for now
    const result = await searchEngine(query, videoId);

    // Save to history (optional: you can skip userId if not logged in)
    await saveSearchHistory("guest", videoId, query, result);

    // Redirect to result page
    res.redirect(`/search/result?videoId=${videoId}&segment=${JSON.stringify(result.bestMatch)}`);
  } catch (error) {
    console.error('Search error:', error);
    res.status(500).send('Search error');
  }
});


module.exports = router;
