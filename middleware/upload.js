const multer = require("multer");
const path = require("path");
const fs = require("fs");

// 1. Creates uploads directory if it doesn't exist.
const uploadDir = "public/uploads/profiles";
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

// 2. Configures multer storage for profile picture uploads.
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "uploads/profiles");
  },
  filename: function (req, file, cb) {
    const uniqueName = Date.now() + "-" + file.originalname;
    cb(null, uniqueName);
  },
});

// 3. Limits uploads to image file types only.
const upload = multer({
  storage,
  fileFilter: function (req, file, cb) {
    // Only allow image files
    const ext = path.extname(file.originalname).toLowerCase();
    if (![".jpg", ".jpeg", ".png", ".gif"].includes(ext)) {
      return cb(new Error("Only image files are allowed"));
    }
    cb(null, true);
  },
});

module.exports = upload;
