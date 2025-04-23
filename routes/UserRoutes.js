const express = require("express");
const passport = require("passport");
const path = require("path");
const multer = require("multer");
const crypto = require("crypto");
const router = express.Router();
const { ensureAuthenticated } = require("../middleware/auth");
const userController = require("../controllers/UserController");
const User = require("../models/User");

const {
  registerUser,
  loginUser,
  logoutUser,
  getUserProfile,
  updateProfile,
  updateProfilePicture,
  changePassword,
  deleteAccount,
  forgotPassword,
  showResetPasswordForm,
  resetPassword
} = userController;

// ========== Multer Config for Profile Picture Upload ==========
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "uploads/profiles");
  },
  filename: function (req, file, cb) {
    const ext = path.extname(file.originalname);
    const fileName = `${Date.now()}-${file.originalname}`;
    cb(null, fileName);
  }
});
const upload = multer({ storage });

// ========== AUTH ========== //
router.post("/register", registerUser);

router.post("/login", (req, res, next) => {
  passport.authenticate("local", (err, user, info) => {
    if (err) return next(err);

    if (!user) {
      return res.status(401).render("login", {
        error: "Invalid email or password.",
        showPopup: true,
        email: req.body.email
      });
    }
    

    req.logIn(user, (err) => {
      if (err) return next(err);

      req.session.user = {
        _id: user._id,
        name: user.name,
        email: user.email,
        profilePicture: user.profilePicture,
        isGoogleUser: user.isGoogleUser
      };

      return res.redirect("/");
    });
  })(req, res, next);
});

router.post("/logout", (req, res) => {
  req.session.destroy(() => {
    res.redirect("/");
  });
});

// ========== PROFILE ROUTES ========== //
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

// ========== DELETE ACCOUNT ==========
router.post("/delete-account", ensureAuthenticated, deleteAccount);

// ========== FORGOT PASSWORD ==========
router.get("/forgot-password", (req, res) => {
  const origin = req.query.from === "profile" ? "profile" : "public";
  res.render("forgot-password", {
    error: null,
    success: null,
    user: req.user || req.session.user || null,
    origin,
  });
});
router.get('/check-provider', async (req, res) => {
  const email = req.query.email;
  try {
    const user = await User.findOne({ email });
    if (!user) return res.json({ provider: null });

    return res.json({
      provider: user.isGoogleUser ? 'google' : 'local'
    });

  } catch (err) {
    return res.status(500).json({ error: 'Server error' });
  }
});
router.post("/forgot-password", forgotPassword);
router.get('/reset-password/:token', async (req, res) => {
  try {
    const rawToken = req.params.token;
    const hashedToken = crypto.createHash("sha256").update(rawToken).digest("hex");

const user = await User.findOne({ resetPasswordToken: hashedToken });


    if (!user) {
      return res.status(400).render("reset-used"); // token not found at all
    }
    
    if (!user.resetPasswordExpires || user.resetPasswordExpires < Date.now()) {
      return res.status(400).render("reset-expired"); // token is found but expired
    }
    

    res.render('reset-password', { token: rawToken });
  } catch (error) {
    console.error(error);
    res.status(500).send('Server error');
  }
});
router.post('/reset-password/:token', resetPassword);
router.get("/check-email", async (req, res) => {
  const email = req.query.email;
  if (!email) return res.json({ available: false });

  const user = await User.findOne({ email });
  return res.json({ available: !user });
});
// ========== GOOGLE AUTH ==========
router.get("/auth/google", passport.authenticate("google", { scope: ["profile", "email"] }));

router.get(
  "/auth/google/callback",
  passport.authenticate("google", { failureRedirect: "/login" }),
  (req, res) => res.redirect("/")
);

module.exports = router;
