const express = require("express");
const passport = require("passport");
const path = require("path");
const multer = require("multer");
const router = express.Router();
const { ensureAuthenticated } = require("../middleware/auth");
const userController = require("../controllers/UserController");
const User = require("../models/User");
const userRepository = require("../repositories/userRepository");

const {
  registerUser,
  loginUser,
  logoutUser,
  updateProfile,
  updateProfilePicture,
  changePassword,
  deleteAccount,
  forgotPassword,
  showResetPasswordForm,
  resetPassword,
} = userController;

// Multer Config for Profile Picture Upload.
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "uploads/profiles");
  },
  filename: function (req, file, cb) {
    const ext = path.extname(file.originalname);
    const fileName = `${Date.now()}-${file.originalname}`;
    cb(null, fileName);
  },
});
const upload = multer({ storage });

// 1. Register Route.
router.post("/register", registerUser);

// 2. Login Route.
router.post("/login", loginUser);

// 3. Logout Route.
router.post("/logout", logoutUser);

// 4. Profile Routes.
router.post(
  "/profile/update",
  ensureAuthenticated,
  upload.single("profilePicture"),
  updateProfile
);
router.post(
  "/profile/upload-picture",
  ensureAuthenticated,
  upload.single("profilePicture"),
  updateProfilePicture
);
router.post("/profile/change-password", ensureAuthenticated, changePassword);
router.get("/login", (req, res) => {
  res.render("login", { email: "", error: null, success: null });
});

// 5. Delete Account Route.
router.post("/delete-account", ensureAuthenticated, deleteAccount);

// 6. Forgot Password and Reset Password Routes.
router.get("/forgot-password", (req, res) => {
  const origin = req.query.from === "profile" ? "profile" : "public";
  res.render("forgot-password", {
    error: null,
    success: null,
    user: req.user || req.session.user || null,
    origin,
  });
});
router.get("/check-provider", async (req, res) => {
  const email = req.query.email;
  try {
    const user = await userRepository.findByEmail(email);
    if (!user) return res.json({ provider: null });
    return res.json({
      provider: user.isGoogleUser ? "google" : "local",
    });
  } catch (err) {
    return res.status(500).json({ error: "Server error" });
  }
});
router.post("/forgot-password", forgotPassword);
router.get("/reset-password/:token", showResetPasswordForm);
router.post("/reset-password/:token", resetPassword);
router.get("/check-email", async (req, res) => {
  const email = req.query.email;
  if (!email) return res.json({ available: false });

  const user = await userRepository.findByEmail(email);
  return res.json({ available: !user });
});

// 7. Google Auth Routes.
router.get(
  "/auth/google",
  passport.authenticate("google", { scope: ["profile", "email"] })
);
router.get(
  "/auth/google/callback",
  passport.authenticate("google", { failureRedirect: "/login" }),
  (req, res) => res.redirect("/")
);

module.exports = router;
