const Video = require("../models/Video");
const User = require("../models/User");
const Report = require("../models/Report");

const supabase = require("../supabaseClient");
const fs = require("fs");
const path = require("path");

exports.getAdminDashboard = async (req, res) => {
  try {
    const videos = await Video.find({ isHidden: { $ne: true } }).populate("user");

    // Download videos locally if not present
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

    const reports = await Report.find()
      .populate({ path: "video", populate: { path: "user" } }) // this populates uploader
      .populate("reportedBy");


    const reportedVideoIds = [...new Set(reports.map(r => r.video?._id.toString()))];
    const reportedUserIds = [...new Set(reports.map(r => r.reportedBy?._id.toString()))];

    // Calculate counts before rendering
    const pendingReportsCount = await Report.countDocuments({ status: "pending" });
    const resolvedReportsCount = await Report.countDocuments({ status: "resolved" });

    // Single render call passing all variables
    res.render("admin", {
      videos, users, reports, reportedVideoIds, reportedUserIds,
      pendingReportsCount,
      resolvedReportsCount
    });

  } catch (err) {
    console.error("Error loading admin dashboard:", err);
    res.status(500).send("Server Error");
  }
};


exports.deleteVideo = async (req, res) => {
  try {
    const videoId = req.params.id;
    await Video.findByIdAndDelete(videoId);
    res.status(200).json({ message: "Video deleted successfully" });
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

exports.getReports = async (req, res) => {
  try {
    res.redirect("/admin#reports");
  } catch (error) {
    console.error("Get reports error:", error);
    res.redirect("/admin");
  }
};


exports.deleteReport = async (req, res) => {
  try {
    await Report.findByIdAndDelete(req.params.reportId);
    req.session.toast = { message: "Report deleted.", type: "success" };
    res.redirect("/admin#reports");
  } catch (error) {
    console.error("Delete report error:", error);
    req.session.toast = { message: "Failed to delete report.", type: "error" };
    res.redirect("/admin#reports");
  }
};
exports.resolveReport = async (req, res) => {
  try {
    console.log("Report ID received:", req.params.reportId);
    const report = await Report.findById(req.params.reportId);
    if (!report) {
      req.session.toast = { message: "Report not found.", type: "error" };
      return res.redirect("/admin#reports");
    }

    report.status = "resolved";
    await report.save();

    req.session.toast = { message: "Report marked as resolved.", type: "success" };
    res.redirect("/admin#reports");
  } catch (err) {
    console.error("Error resolving report:", err);
    req.session.toast = { message: "Failed to update report.", type: "error" };
    res.redirect("/admin#reports");
  }
};
