require("dotenv").config();

// Core Modules & Middleware
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
require("events").EventEmitter.defaultMaxListeners = 20;

// Auth Middleware
const { ensureAuthenticated } = require("./middleware/auth");
require("./middleware/auth");

// Models
const Video = require("./models/Video");
const User = require("./models/User");
const ShotMetadata = require("./models/ShotData");
const UserQuery = require("./models/UserQuery");
const ResultVideo = require("./models/ResultVideo");
const SceneMetadata = require("./models/SceneMetadata");
const SceneResults = require("./models/SceneSearchResult");


// DB Connection
const connectDB = require("./config/db");
connectDB();

// Express Setup
const app = express();
const PORT = process.env.PORT || 5000;

// Upload Directory
const uploadDir = path.join(__dirname, "uploads", "profiles");
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

// Middleware
app.use(express.urlencoded({ extended: true }));
app.use(express.json());
app.use(cors());
app.use(morgan("dev"));
app.use("/uploads", express.static(path.join(__dirname, "uploads")));
app.use("/output/clips", express.static(path.join(__dirname, "output/clips")));
app.use(express.static(path.join(__dirname, "public")));

// Session and Passport Setup
app.use(
  session({
    secret: process.env.SESSION_SECRET || "your_secret_key",
    resave: false,
    saveUninitialized: false,
  })
);
app.use(passport.initialize());
app.use(passport.session());

// Attach User Middleware
const attachUser = require("./middleware/attachUser");
app.use(attachUser);

// View Engine
app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "views"));

// Global Locals Middleware
app.use((req, res, next) => {
  const user = req.user || req.session.user;
  res.locals.user = user || null;
  res.locals.session = req.session;
  res.locals.isBanned = false;
  if (res.locals.user && res.locals.user.banUntil) {
    const banUntil = new Date(res.locals.user.banUntil);
    if (banUntil > new Date()) {
      res.locals.isBanned = true;
    }
  }
  res.locals.session = req.session;
  if (req.session.toast) {
    res.locals.toast = req.session.toast;
    delete req.session.toast;
  }
  next();
});

// UI Routes
app.get("/", (req, res) => {
  const resetParam = req.query.reset;
  if (resetParam === "success" && !req.session.resetSuccess) {
    return res.status(403).render("access-denied");
  }
  const reset = req.session.resetSuccess ? "success" : null;
  req.session.resetSuccess = false;
  res.render("index", {
    user: req.user || req.session.user,
    reset: reset,
  });
});
app.get("/login", (req, res) => {
  const bannedUntil = req.query.banned;
  res.render("login", { bannedUntil });
});
app.get("/register", (req, res) => res.render("register"));
app.get("/upload", ensureAuthenticated, (req, res) => res.render("upload"));
const searchController = require("./controllers/SearchController");
app.get("/search", ensureAuthenticated, searchController.renderSearchPage);
app.get("/results", ensureAuthenticated, (req, res) => {
  const results = req.session.searchResults || [];
  const query = req.session.searchQuery || "";
  const videoTitle = req.session.videoTitle || "unknown";

  res.render("results", {
    results,
    query,
    video: `/uploads/dl_${videoTitle}.mp4`,
    message: results.length === 0 ? "No results matched your query." : null,
  });
});

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
  const [firstName, lastName] = (req.session.user.name || "").split(" ");

  if (user.banUntil && user.banUntil > new Date()) {
    const banUntilDate = user.banUntil.toLocaleDateString();
    return res.redirect(`/login?banned=${encodeURIComponent(banUntilDate)}`);
  }
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

// Password Reset Routes
app.post("/set-theme", (req, res) => {
  const { theme } = req.body;
  req.session.theme = theme;
  res.sendStatus(200);
});
app.post("/reset-password/:token", async (req, res) => {
  try {
    const { password, confirmPassword } = req.body;
    const token = req.params.token;
    const hashedToken = crypto.createHash("sha256").update(token).digest("hex");
    const user = await User.findOne({ resetPasswordToken: hashedToken });
    if (!user) return res.status(400).render("reset-used");
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
    user.password = await bcrypt.hash(password, 10);
    user.resetPasswordToken = undefined;
    user.resetPasswordExpires = undefined;
    await user.save();
    req.session.resetSuccess = true;
    req.session.regenerate((err) => {
      if (err) {
        console.error("Error regenerating session:", err);
        return res.status(500).send("Error logging out.");
      }
      req.session.resetSuccess = true;
      return res.redirect("/");
    });
  } catch (err) {
    console.error("Reset error:", err);
    return res.status(500).send("Server error");
  }
});

// Authentication - Google OAuth
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
      if (user.banUntil && user.banUntil > new Date()) {
        const banUntilDate = user.banUntil.toLocaleDateString();
        return res.redirect(`/login?banned=${encodeURIComponent(banUntilDate)}`);
      }

      req.session.user = {
        _id: user._id,
        name: user.name,
        email: user.email,
        profilePicture: user.profilePicture || null,
        isGoogleUser: user.isGoogleUser,
        isAdmin: user.isAdmin || false,
      };

      res.redirect("/");

    } catch (err) {
      console.error("Google login error:", err.message);
      res.redirect("/login");
    }
  }
);

// Multer Config for File Uploads
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

// Route Imports and Mounts
const uploadRoutes = require("./routes/UploadRoutes");
const userRoutes = require("./routes/UserRoutes");
const videoRoutes = require("./routes/VideoRoutes");
const historyRoutes = require("./routes/HistoryRoutes");
const searchRoutes = require("./routes/SearchRoutes");
const catalogRoutes = require("./routes/CatalogRoutes");
const adminRoutes = require('./routes/AdminRoutes');

app.use("/api/videos", videoRoutes);
app.use("/users", userRoutes);
app.use("/", historyRoutes);
app.use("/api/search", searchRoutes);
app.use("/", uploadRoutes);
app.use("/", catalogRoutes);
app.use('/', adminRoutes);


// 404 Fallback
app.use("*", (req, res) => {
  res.status(404).render("error", {
    error: { status: 404, message: "Page Not Found" },
    theme: req.session.theme || "light",
  });
});

// Start Server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
