const path = require('path');
const { exec } = require('child_process');

exports.showResult = async (req, res) => {
  try {
    const { videoId, segment } = req.query;
    const { start, end } = JSON.parse(segment);

    const inputPath = path.join(__dirname, '../uploads', `${videoId}.mp4`);
    const outputFile = `${videoId}_${start.replace(/:/g, '-')}_${end.replace(/:/g, '-')}.mp4`;
    const outputPath = path.join(__dirname, '../public/output/clips', outputFile);

    const command = `ffmpeg -i "${inputPath}" -ss ${start} -to ${end} -c copy "${outputPath}"`;

    exec(command, (err) => {
      if (err) {
        console.error('FFmpeg error:', err);
        return res.status(500).send('Error generating video clip');
      }

      // Render results.ejs and pass the video clip path
      res.render('results', { videoPath: `/output/clips/${outputFile}` });
    });
  } catch (error) {
    console.error('Result generation error:', error);
    res.status(500).send('Server error');
  }
};
