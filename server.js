require("dotenv").config();
const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const morgan = require("morgan");
const path = require("path");
const multer = require("multer");
const { exec } = require("child_process");
const fs = require("fs");
const axios = require("axios");
const session = require("express-session");
const passport = require("passport");
const bcrypt = require("bcryptjs");
const crypto = require("crypto");
const { ensureAuthenticated } = require("./middleware/auth");
require("./middleware/auth");
require("events").EventEmitter.defaultMaxListeners = 20;
const Video = require("./models/Video");

const { getProfile } = require("./controllers/UserController");

const User = require("./models/User");
const supabase = require("./supabaseClient");
const ShotMetadata = require("./models/ShotData");

const UserQuery = require("./models/UserQuery");
const ResultVideo = require("./models/ResultVideo");

const SceneMetadata = require("./models/SceneMetadata");
const SceneResults = require("./models/SceneSearchResult");

const connectDB = require("./config/db");
const videoRoutes = require("./routes/VideoRoutes");
const userRoutes = require("./routes/UserRoutes");
const historyRoutes = require("./routes/HistoryRoutes");
const searchRoutes = require("./routes/SearchRoutes");
const uploadDir = path.join(__dirname, "uploads", "profiles");

// ✅ Ensure upload directory exists
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

const app = express();
const PORT = process.env.PORT || 5000;

// Connect DB
connectDB();

// Middleware
app.use(express.urlencoded({ extended: true }));
app.use("/uploads", express.static(path.join(__dirname, "uploads")));
app.use(express.json());
app.use(cors());
app.use(morgan("dev"));
app.use(express.static(path.join(__dirname, "public")));

// Session and Passport
// Session and Passport
app.use(
  session({
    secret: process.env.SESSION_SECRET || "your_secret_key",
    resave: false,
    saveUninitialized: false,
  })
);
app.use(passport.initialize());
app.use(passport.session());

// ✅ Attach user after session + passport are ready
const attachUser = require("./middleware/attachUser");
app.use(attachUser);

// Then your global middleware
app.use((req, res, next) => {
  res.locals.user = req.user || req.session.user || null;
  res.locals.session = req.session;
  if (req.session.toast) {
    res.locals.toast = req.session.toast;
    delete req.session.toast;
  }
  next();
});

// View Engine
app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "views"));

// UI Pages
app.get("/", (req, res) => {
  const resetParam = req.query.reset;

  // ❌ If someone adds ?reset=success but no session flag set, block them
  if (resetParam === "success" && !req.session.resetSuccess) {
    return res.status(403).render("access-denied"); // or "404" if you prefer
  }

  const reset = req.session.resetSuccess ? "success" : null;
  req.session.resetSuccess = false;

  res.render("index", {
    user: req.user || req.session.user,
    reset: reset,
  });
});

app.get("/login", (req, res) => res.render("login"));
app.get("/register", (req, res) => res.render("register"));
app.get("/upload", ensureAuthenticated, (req, res) => res.render("upload"));
app.get("/search", ensureAuthenticated, (req, res) => {
  res.render("search", {
    videoId: req.query.videoId || "",
    videoFilename: null,
    videoTitle: null,
  });
});

app.get("/results", ensureAuthenticated, (req, res) => res.render("results"));
app.get("/scene-history", ensureAuthenticated, (req, res) =>
  res.render("scene-history")
);
app.get("/forgot-password", (req, res) => {
  res.render("forgot-password", {
    user: req.session.user || null,
  });
});

app.get("/profile", ensureAuthenticated, async (req, res) => {
  const userId = req.session.user?._id || req.user?._id;
  if (!userId) return res.redirect("/login");

  const user = await User.findById(userId);

  if (req.user) {
    req.user.name = user.name;
    req.user.email = user.email;
    req.user.profilePicture = user.profilePicture;
    req.user.isGoogleUser = user.isGoogleUser;
  }

  const success = req.session.success;
  delete req.session.success;

  // ✅ Fix here
  const [firstName, lastName] = (req.session.user.name || "").split(" ");

  res.render("profile", {
    user: req.session.user,
    firstName,
    lastName,
    success,
    formError: null,
    mismatch: null,
    incorrectPassword: null,
    weakPassword: null,
  });
});

// RESET PASSWORD PAGE

app.post("/set-theme", (req, res) => {
  const { theme } = req.body;
  req.session.theme = theme;
  res.sendStatus(200); // or res.json({ success: true });
});

// ✅ RESET PASSWORD SUBMIT & LOGIN
app.post("/reset-password/:token", async (req, res) => {
  try {
    const { password, confirmPassword } = req.body;
    const token = req.params.token;
    const hashedToken = crypto.createHash("sha256").update(token).digest("hex");

    const user = await User.findOne({ resetPasswordToken: hashedToken });

    if (!user) {
      return res.status(400).render("reset-used");
    }

    if (!user.resetPasswordExpires || user.resetPasswordExpires < Date.now()) {
      return res.status(400).render("reset-expired");
    }

    const passwordRegex = /^(?=.*[A-Za-z])(?=.*[^A-Za-z0-9]).{9,}$/;
    if (!passwordRegex.test(password)) {
      return res.render("reset-password", {
        token,
        error:
          "Password must be at least 9 characters and contain a letter and a special character.",
        success: null,
      });
    }

    if (password !== confirmPassword) {
      return res.render("reset-password", {
        token,
        error: "Passwords do not match.",
        success: null,
      });
    }

    // ✅ Update password
    user.password = await bcrypt.hash(password, 10);

    // ✅ Invalidate the reset token
    user.resetPasswordToken = undefined;
    user.resetPasswordExpires = undefined;

    await user.save();

    // ✅ Log the user out by destroying the session
    req.session.resetSuccess = true;

    // Destroy session securely
    req.session.regenerate((err) => {
      if (err) {
        console.error("Error regenerating session:", err);
        return res.status(500).send("Error logging out.");
      }
      // ✅ Set success flag on new session
      req.session.resetSuccess = true;
      return res.redirect("/");
    });
  } catch (err) {
    console.error("Reset error:", err);
    return res.status(500).send("Server error");
  }
});

// Google OAuth
app.get(
  "/auth/google",
  passport.authenticate("google", { scope: ["profile", "email"] })
);
app.get(
  "/auth/google/callback",
  passport.authenticate("google", { failureRedirect: "/login" }),
  async (req, res) => {
    try {
      const profile = req.user;
      let user = await User.findOne({ email: profile.email });

      if (!user) {
        user = new User({
          name: profile.name,
          email: profile.email,
          isGoogleUser: true,
        });
        await user.save();
      }

      req.session.user = {
        _id: user._id,
        name: user.name,
        email: user.email,
        profilePicture: user.profilePicture || null,
        isGoogleUser: user.isGoogleUser,
      };

      res.redirect("/");
    } catch (err) {
      console.error("Google login error:", err.message);
      res.redirect("/login");
    }
  }
);

// Multer Config
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "uploads/");
  },
  filename: function (req, file, cb) {
    const ext = path.extname(file.originalname);
    const rawTitle = req.body.title || "untitled";
    const safeTitle = rawTitle.trim().replace(/[^a-z0-9_\-]/gi, "_");
    cb(null, `${safeTitle}${ext}`);
  },
});
const upload = multer({ storage });

// APIs
app.use("/api/videos", videoRoutes);
app.use("/api/search", searchRoutes);
app.use("/", historyRoutes);
app.use("/", userRoutes);
app.use("/users", userRoutes);
app.use("/output", express.static(path.join(__dirname, "output")));
// Start Server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

app.post("/search/user", ensureAuthenticated, async (req, res) => {
  const { query } = req.body;
  const userId = req.user?._id || req.session.user?._id;

  // ✅ Get videoTitle from session
  const safeTitle = req.session.videoTitle;

  if (!safeTitle) {
    return res.status(400).send("No video linked to this search.");
  }

  try {
    const newQuery = new UserQuery({ userId, query });
    await newQuery.save();

    const captionsPath = path.join(__dirname, `${safeTitle}_captions.json`);
    const videoPath = path.join("uploads", `dl_${safeTitle}.mp4`);

    if (!fs.existsSync(captionsPath) || !fs.existsSync(videoPath)) {
      return res.status(400).render("error", {
        error: { status: 400, message: "Missing captions or video." },
        theme: req.session.theme || "light",
        friendlyMessage:
          "The video or its subtitles couldn't be found. Please re-upload your video.",
      });
    }

    const pyCommand = `set PYTHONPATH=. && venv_search\\Scripts\\python.exe AI/search/search_user_query.py "${captionsPath}" "${query}" "${videoPath}"`;

    exec(pyCommand, async (error, stdout, stderr) => {
      if (error) {
        console.error("❌ Python search failed:", error.message);
        console.error("🔴 STDERR:", stderr);
        return res.status(500).render("error", {
          error: { status: 500, message: "Search processing failed." },
          theme: req.session.theme || "light",
          friendlyMessage:
            "There was a problem analyzing your search. Please try again later.",
        });
      }

      console.log("✅ Search finished:", stdout);
      res.redirect("/search-results");
    });
  } catch (err) {
    console.error("❌ Error saving query or executing search:", err.message);
    return res.status(500).render("error", {
      error: { status: 500, message: "Unexpected server error." },
      theme: req.session.theme || "light",
      friendlyMessage:
        "Something went wrong on our end. Please try again shortly.",
    });
  }
});

const uploadRoutes = require("./routes/UploadRoutes");
app.use("/", uploadRoutes);

app.post("/search/result", ensureAuthenticated, async (req, res) => {
  const userQuery = req.body.query;
  let videoTitle;
  if (req.body.videoId) {
    const video = await Video.findById(req.body.videoId);
    videoTitle = video?.filename.replace(/^dl_/, "").replace(/\.mp4$/, "");
  } else {
    videoTitle = req.session.videoTitle || req.body.videoTitle || null;
  }

  req.session.videoTitle = videoTitle;

  if (!videoTitle || !userQuery) {
    return res.status(400).render("error", {
      error: {
        status: 400,
        message: "Missing video or search input",
      },
      theme: req.session.theme || "light",
      friendlyMessage:
        "Oops! We couldn’t find your video or search term. Please try uploading a video again or go back to the search page.",
    });
  }

  const captionPath = path.join(__dirname, `${videoTitle}_captions.json`);

  if (!fs.existsSync(captionPath)) {
    return res.status(404).send("Caption data not found.");
  }

  try {
    const rawData = fs.readFileSync(captionPath, "utf-8").trim();

    if (rawData.startsWith("<")) {
      console.error("❌ Captions file contains HTML, not JSON");
      return res.status(500).render("error", {
        error: { status: 500, message: "Corrupted captions file" },
        theme: req.session.theme || "light",
        friendlyMessage:
          "Something went wrong while reading the video data. Please try re-uploading your video.",
      });
    }

    let parsed;
    try {
      parsed = JSON.parse(rawData);
    } catch (e) {
      console.error("❌ Failed to parse captions JSON:", e.message);
      return res.status(500).render("error", {
        error: { status: 500, message: "Invalid JSON format" },
        theme: req.session.theme || "light",
        friendlyMessage:
          "We couldn’t process your video results. Please try again later.",
      });
    }

    const metadata = parsed.shots_metadata;

    const results = [];

    for (const [imgPath, shot] of Object.entries(metadata)) {
      if (shot.caption.toLowerCase().includes(userQuery.toLowerCase())) {
        results.push({
          caption: shot.caption,
          tags: shot.tags,
          start_time: shot.start_time,
          end_time: shot.end_time,
          image: imgPath,
        });
      }
    }

    if (results.length === 0) {
      return res.render("results", {
        results: [],
        query: userQuery,
        video: `/uploads/dl_${videoTitle}.mp4`,
        message: "No results matched your query.",
      });
    }

    const { execSync } = require("child_process");

    const inputPath = path.join("uploads", `dl_${videoTitle}.mp4`);
    const clipsDir = path.join("output", "clips");
    if (!fs.existsSync(clipsDir)) {
      fs.mkdirSync(clipsDir, { recursive: true });
    }

    const trimmedResults = [];

    for (const match of results) {
      const clipName = `${videoTitle}_${match.start_time.replace(
        /:/g,
        "-"
      )}_${match.end_time.replace(/:/g, "-")}_clip.mp4`;
      const outputClipPath = path.join(clipsDir, clipName);
      const clipUrl = `/output/clips/${clipName}`;

      if (!fs.existsSync(outputClipPath)) {
        const trimCommand = `ffmpeg -y -i "${inputPath}" -ss ${match.start_time} -to ${match.end_time} -preset ultrafast -crf 28 "${outputClipPath}"`;
        try {
          execSync(trimCommand);
          console.log("✅ Created clip:", clipName);
        } catch (error) {
          console.error("❌ Failed to trim clip:", error.message);
          continue; // Skip this one if failed
        }
      }

      trimmedResults.push({
        ...match,
        clip: clipUrl,
      });
    }

    const clipToSave = trimmedResults[0]?.clip || null;

    if (trimmedResults.length > 0 && req.session.user) {
      try {
        // Save the user query
        const newQuery = await UserQuery.create({
          userId: req.session.user._id,
          videoId: req.body.videoId,
          query: userQuery,
        });

        // Save each result video
        for (const match of trimmedResults) {
          const clipFilename = match.clip.split("/").pop();
          await ResultVideo.create({
            queryId: newQuery._id,
            clipFilename,
            timeRange: `${match.start_time} - ${match.end_time}`,
            caption: match.caption,
          });
        }

        console.log("✅ Query and results saved to history.");
      } catch (err) {
        console.warn("⚠️ Failed to save query/results:", err.message);
      }
    }
    res.render("results", {
      results: trimmedResults,
      query: userQuery,
      video: null,
      message: null,
    });
  } catch (err) {
    console.error("❌ Failed to process search results:", err.message);
    return res.status(500).send("Server error processing results.");
  }
});
app.use("*", (req, res) => {
  res.status(404).render("error", {
    error: { status: 404, message: "Page Not Found" },
    theme: req.session.theme || "light",
  });
});
