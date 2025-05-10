const express = require("express");
const router = express.Router();
const multer = require("multer");
const path = require("path");
const { exec } = require('child_process');
const { uploadVideo, getAllVideos, getVideoById } = require("../controllers/VideoController");
const { searchEngine, saveSearchHistory } = require('../utils/searchEngine');

const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, "uploads/"),
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname);
    const safeTitle = req.body.title?.replace(/[^a-z0-9_\-]/gi, "_") || "untitled";
    cb(null, `${safeTitle}${ext}`);
  },
});
const upload = multer({ storage });

// Updated route
router.post('/result', async (req, res) => {
  const { query, videoId } = req.body;

  try {
    // Call your search AI model (dummy logic for now)
    const result = await searchEngine(query, videoId);

    // Save search query to DB (for history)
    await saveSearchHistory(req.user?.id || "guest", videoId, query, result);

    // Redirect to result page with video + timestamps
    res.redirect(`/search/result?videoId=${videoId}&segment=${JSON.stringify(result.bestMatch)}`);
  } catch (error) {
    console.error(error);
    res.status(500).send('Search error');
  }
});


router.get('/result', async (req, res) => {
  const { videoId, segment } = req.query;
  const { start, end } = JSON.parse(segment);

  const inputPath = path.join(__dirname, '../uploads', `${videoId}.mp4`);
  const outputFile = `${videoId}_${start.replace(/:/g, '-')}_${end.replace(/:/g, '-')}.mp4`;
  const outputPath = path.join(__dirname, '../output/clips', outputFile);

  const command = `ffmpeg -i "${inputPath}" -ss ${start} -to ${end} -c copy "${outputPath}"`;

  exec(command, (err) => {
    if (err) {
      console.error('FFmpeg error:', err);
      return res.status(500).send('Error generating clip');
    }

    res.render('results', { videoPath: `/output/clips/${outputFile}` });
  });
});

router.get("/:id", getVideoById);

module.exports = router;
