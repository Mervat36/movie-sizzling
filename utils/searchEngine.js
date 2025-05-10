// utils/searchEngine.js

const fs = require('fs');
const path = require('path');

// Fake AI search function for now (replace with real model logic)
exports.searchEngine = async (query, videoId) => {
  console.log(`🔍 Running AI search for query: "${query}" in video ID: ${videoId}`);

  // Simulated best match time
  return {
    bestMatch: {
      start: '00:01:05',
      end: '00:01:45',
      query: query
    }
  };
};

// Save search history to JSON (or connect to MongoDB/Supabase instead)
exports.saveSearchHistory = async (userId, videoId, query, result) => {
  const historyFile = path.join(__dirname, '../data/search_history.json');

  const historyEntry = {
    userId,
    videoId,
    query,
    start: result.bestMatch.start,
    end: result.bestMatch.end,
    timestamp: new Date().toISOString()
  };

  let historyData = [];

  try {
    if (fs.existsSync(historyFile)) {
      const raw = fs.readFileSync(historyFile);
      historyData = JSON.parse(raw);
    }
    historyData.push(historyEntry);
    fs.writeFileSync(historyFile, JSON.stringify(historyData, null, 2));
    console.log("✅ Search history saved.");
  } catch (err) {
    console.error("❌ Failed to save search history:", err);
  }
};
