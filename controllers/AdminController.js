const Video = require("../models/Video");
const User = require("../models/User");

const supabase = require("../supabaseClient");
const fs = require("fs");
const path = require("path");

exports.getAdminDashboard = async (req, res) => {
  try {
    const videos = await Video.find({ isHidden: { $ne: true } }).populate("user");

    // Make sure videos are downloaded from Supabase if not local
    await Promise.all(videos.map(async (video) => {
      const localPath = path.join("uploads", video.filename);
      if (!fs.existsSync(localPath)) {
        const actualFilename = video.filename.replace(/^dl_/, "");
        const { data, error } = await supabase.storage.from("movies").download(actualFilename);
        if (data) {
          const buffer = Buffer.from(await data.arrayBuffer());
          fs.writeFileSync(localPath, buffer);
        } else {
          console.error(`❌ Failed to fetch video ${video.filename} from Supabase:`, error?.message);
        }
      }
    }));

    const users = await User.find();
    res.render("admin", { videos, users });
  } catch (err) {
    console.error("Error loading admin dashboard:", err);
    res.status(500).send("Server Error");
  }
};



exports.deleteVideo = async (req, res) => {
  try {
    const videoId = req.params.id;
    await Video.findByIdAndDelete(videoId);
    res.redirect("/admin");
  } catch (err) {
    console.error("Error deleting video:", err);
    res.status(500).send("Server Error");
  }
};

exports.makeAdmin = async (req, res) => {
  try {
    const user = await User.findById(req.params.id);
    if (user) {
      user.isAdmin = true;
      await user.save();
    }
    res.redirect("/admin");
  } catch (err) {
    console.error("Error promoting user:", err);
    res.status(500).send("Server Error");
  }
};

exports.banUser = async (req, res) => {
  try {
    const { banUntil } = req.body;
    const user = await User.findById(req.params.id);
    if (user && banUntil) {
      user.banUntil = new Date(banUntil);
      await user.save();
    }
    res.status(200).json({ message: "User banned" });
  } catch (err) {
    console.error("Error banning user:", err);
    res.status(500).send("Server Error");
  }
};

exports.deleteUser = async (req, res) => {
  try {
    const userId = req.params.id;
    await User.findByIdAndDelete(userId);
    await Video.deleteMany({ user: userId });
    res.redirect("/admin");
  } catch (err) {
    console.error("Error deleting user:", err);
    res.status(500).send("Server Error");
  }
};
exports.unbanUser = async (req, res) => {
  try {
    const user = await User.findById(req.params.id);
    if (user) {
      user.banUntil = null;
      await user.save();
    }
    res.redirect("/admin");
  } catch (err) {
    console.error("Error unbanning user:", err);
    res.status(500).send("Server Error");
  }
};
exports.removeAdmin = async (req, res) => {
  try {
    const user = await User.findById(req.params.id);
    if (user) {
      user.isAdmin = false;
      await user.save();
    }
    res.redirect("/admin");
  } catch (err) {
    console.error("Error removing admin:", err);
    res.status(500).send("Server Error");
  }
};
