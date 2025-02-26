require("dotenv").config();
const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const morgan = require("morgan");
const path = require("path");

// Import Routes
const videoRoutes = require("./routes/videoRoutes");
const userRoutes = require("./routes/userRoutes");
const searchRoutes = require("./routes/searchRoutes");

// Import Database Connection
const connectDB = require("./config/db");

// Initialize App
const app = express();
const PORT = process.env.PORT || 5000;

// Connect to Database
connectDB();

// Middleware
app.use(express.json());
app.use(cors());
app.use(morgan("dev"));
app.use(express.static(path.join(__dirname, "public")));

// Set View Engine
app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "app/views"));

// Routes
app.use("/api/videos", videoRoutes);
app.use("/api/users", userRoutes);
app.use("/api/search", searchRoutes);

// View Routes
app.get("/", (req, res) => res.render("index"));
app.get("/login", (req, res) => res.render("login"));
app.get("/register", (req, res) => res.render("register"));
app.get("/upload", (req, res) => res.render("upload"));
app.get("/search", (req, res) => res.render("search"));
app.get("/results", (req, res) => res.render("results"));
app.get("/history", (req, res) => res.render("history_page"));
app.get("/scene-history", (req, res) => res.render("scene_history"));
app.get("/forgot-password", (req, res) => res.render("forgot-password"));

// Start Server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
