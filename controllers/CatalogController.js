const Video = require("../models/Video");
const Caption = require("../models/Caption");
const fs = require("fs");
const path = require("path");
const { exec } = require("child_process");
const supabase = require("../supabaseClient");

exports.renderCatalogPage = async (req, res) => {
  try {
    const allVideos = await Video.find().sort({ createdAt: -1 });
    res.render("catalog", { videos: allVideos });
  } catch (err) {
    console.error("❌ Error in catalog:", err.message);
    res.status(500).render("error", {
      error: { status: 500, message: "Failed to load catalog." },
      theme: req.session.theme || "light",
    });
  }
};

exports.handleCatalogClick = async (req, res) => {
  const videoId = req.params.id;
  try {
    const video = await Video.findById(videoId);
    if (!video) return res.status(404).send("Video not found in DB.");

    const safeTitle = video.title.trim().replace(/ /g, "_");
    const localPath = path.join("uploads", video.filename);

    // ✅ Download if not exists
    if (!fs.existsSync(localPath)) {
      console.log("📥 Downloading from Supabase...");
      const { data, error } = await supabase.storage
        .from("movies")
        .download(video.filename.replace(/^dl_/, ""));
      
      if (error || !data) {
        console.error("❌ Supabase download failed:", error?.message);
        return res.status(500).send("Failed to fetch video from cloud.");
      }

      const buffer = Buffer.from(await data.arrayBuffer());
      fs.writeFileSync(localPath, buffer);
      console.log("✅ Download complete.");
    }

    // ✅ Save title in session
    req.session.videoTitle = safeTitle;

    res.redirect(`/search?videoId=${video._id}`);
  } catch (err) {
    console.error("❌ handleCatalogClick error:", err.message);
    res.status(500).send("Catalog click failed.");
  }
};

exports.searchCatalog = async (req, res) => {
  const { query, videoId } = req.body;
  
  if (!query || !videoId) {
    return res.status(400).json({ error: "Query and videoId are required" });
  }

  try {
    // 1. Get video info
    const video = await Video.findById(videoId);
    if (!video) {
      return res.status(404).json({ error: "Video not found" });
    }

    const safeTitle = video.title.trim().replace(/ /g, "_");

    // 2. Get captions from MongoDB
    const captionData = await Caption.findOne({ movie_name: safeTitle });
    if (!captionData) {
      return res.status(404).json({ error: "Captions not found in database" });
    }

    // 3. Create temporary JSON file for Python script
    const tempJsonPath = path.join(__dirname, "..", `temp_${safeTitle}_captions.json`);
    const tempData = {
      movie_name: captionData.movie_name,
      shots_metadata: captionData.shots_metadata
    };
    
    fs.writeFileSync(tempJsonPath, JSON.stringify(tempData, null, 2));
    console.log("📁 Created temporary captions file:", tempJsonPath);

    // 4. Get video path
    const videoPath = path.join("uploads", video.filename);
    if (!fs.existsSync(videoPath)) {
      return res.status(404).json({ error: "Video file not found locally" });
    }

    // 5. Run Python search script
    const pyCommand = `set PYTHONPATH=. && venv_search\\Scripts\\python.exe AI/search/search_user_query.py "${tempJsonPath}" "${query}" "${videoPath}"`;

    exec(pyCommand, async (error, stdout, stderr) => {
      // Clean up temporary file
      if (fs.existsSync(tempJsonPath)) {
        fs.unlinkSync(tempJsonPath);
        console.log("🧹 Cleaned up temporary file");
      }

      if (error) {
        console.error("❌ Python search failed:", error.message);
        console.error("🔴 STDERR:", stderr);
        return res.status(500).json({ error: "Search processing failed" });
      }

      console.log("✅ Search finished:", stdout);

      let parsedResults;
      try {
        if (stdout.trim().startsWith("<html")) {
          throw new Error("Received HTML instead of JSON");
        }
        parsedResults = JSON.parse(stdout);
        console.log("📦 Parsed Results Count:", parsedResults.length);
        
        if (!Array.isArray(parsedResults) || parsedResults.length === 0) {
          return res.json({
            results: [],
            query,
            video: `/uploads/${video.filename}`,
            message: "No relatable scenes were found."
          });
        }

        // 6. Process and return results
        const filteredResults = parsedResults
          .filter((result) => result.score >= 0.4)
          .sort((a, b) => b.score - a.score);

        res.json({
          results: filteredResults,
          query,
          video: `/uploads/${video.filename}`,
          message: null
        });

      } catch (e) {
        console.error("❌ Failed to parse Python output:", e.message);
        res.status(500).json({ error: "Search result parse error" });
      }
    });

  } catch (err) {
    console.error("❌ Error in searchCatalog:", err.message);
    res.status(500).json({ error: "Unexpected server error" });
  }
};
