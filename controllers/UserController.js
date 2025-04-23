const User = require("../models/User");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const path = require("path");
const crypto = require("crypto");
const sendResetEmail = require("../utils/sendResetPasswordEmail");

function isStrongPassword(password) {
  return /^(?=.*[A-Za-z])(?=.*[^A-Za-z0-9])(?=.{9,})/.test(password);
}

const registerUser = async (req, res) => {
  const { firstName, lastName, email, password, confirmPassword } = req.body;

  try {
    const existingUser = await User.findOne({ email });

    if (existingUser) {
      if (existingUser.isGoogleUser) {
        // Allow re-registration with password
        const hashedPassword = await bcrypt.hash(password, 10);

        existingUser.name = `${firstName} ${lastName}`;
        existingUser.password = hashedPassword;
        existingUser.isGoogleUser = false;

        await existingUser.save();

        // Update session with all existing user data
        req.session.user = {
          _id: existingUser._id,
          name: existingUser.name,
          email: existingUser.email,
          profilePicture: existingUser.profilePicture || null,
          isGoogleUser: false,
        };

        return res.redirect("/profile");
      }

      // Normal user already exists
      return res.status(409).render("register", {
        error: "User already exists.",
        firstName,
        lastName,
        email,
      });
    }

    if (password !== confirmPassword) {
      return res.status(400).render("register", {
        error: "Passwords do not match.",
        firstName,
        lastName,
        email,
      });
    }

    if (!isStrongPassword(password)) {
      return res.status(400).render("register", {
        error:
          "Password must be at least 9 characters and contain a letter and a special character.",
        firstName,
        lastName,
        email,
      });
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    const newUser = new User({
      name: `${firstName} ${lastName}`,
      email,
      password: hashedPassword,
      isGoogleUser: false,
    });

    await newUser.save();

    req.session.user = {
      _id: newUser._id,
      name: newUser.name,
      email: newUser.email,
      profilePicture: null,
      isGoogleUser: false,
    };

    req.session.toast = {
      type: "success",
      message: "Account created successfully. Welcome!",
    };
    res.redirect("/");
  } catch (error) {
    console.error("Register error:", error);
    return res.status(500).render("register", {
      error: "Server error. Please try again.",
      firstName,
      lastName,
      email,
    });
  }
};

const loginUser = async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).render("login", {
        error: "Invalid email or password.",
        showPopup: true,
        email,
      });
    }

    const user = await User.findOne({ email });

    if (!user || !(await bcrypt.compare(password, user.password))) {
      return res.status(401).render("login", {
        error: "Invalid email or password.",
        showPopup: true,
        email,
      });
    }

    req.session.user = {
      _id: user._id,
      name: user.name,
      email: user.email,
      profilePicture: user.profilePicture || null,
      isGoogleUser: user.isGoogleUser || false,
    };

    res.redirect("/");
  } catch (error) {
    console.error("Login error:", error);
    res.status(500).render("login", {
      error: "Server error occurred. Please try again.",
      showPopup: true,
      email,
    });
  }
};

const logoutUser = (req, res) => {
  req.session.destroy(() => {
    res.redirect("/");
  });
};

const getUserProfile = async (req, res) => {
  try {
    const user = await User.findById(req.params.id).select("-password");
    if (!user) return res.status(404).json({ message: "User not found" });
    res.status(200).json(user);
  } catch (error) {
    console.error("Get profile error:", error);
    res
      .status(500)
      .json({ message: "Error fetching user", error: error.message || error });
  }
};

const getProfile = async (req, res) => {
  try {
    const user = req.session.user;
    if (!user) return res.redirect("/login");

    const success = req.session.success;
    delete req.session.success;

    res.render("profile", {
      user: req.session.user,
      firstName,
      lastName,
      success,
      formError: null,
      mismatch: null,
      incorrectPassword: null,
      weakPassword: null,
      deleteError: null,
    });
  } catch (error) {
    console.error("Profile render error:", error.message);
    res.status(500).send("Server error");
  }
};

const updateProfile = async (req, res) => {
  try {
    const userId = req.session.user._id;
    const user = await User.findById(userId);
    if (!user) return res.status(404).send("User not found.");

    const { firstName, lastName } = req.body;
    if (firstName && lastName) {
      user.name = `${firstName} ${lastName}`;
    }

    if (req.file) {
      user.profilePicture = `/uploads/profiles/${req.file.filename}`;
    }

    await user.save();

    req.session.user = {
      _id: user._id,
      name: user.name,
      email: user.email,
      profilePicture: user.profilePicture || null,
      isGoogleUser: user.isGoogleUser,
    };

    if (req.user) {
      req.user.name = user.name;
      req.user.email = user.email;
      req.user.profilePicture = user.profilePicture;
      req.user.isGoogleUser = user.isGoogleUser;
    }

    req.session.success = "Profile updated successfully.";
    res.redirect("/profile");
  } catch (err) {
    console.error("Update Error:", err.message);
    res.status(500).render("profile", {
      user: req.session.user,
      error: "Something went wrong while updating your profile.",
    });
  }
};

const updateProfilePicture = async (req, res) => {
  try {
    const filePath = "/uploads/profiles/" + req.file.filename;
    const user = await User.findByIdAndUpdate(
      req.session.user._id,
      { profilePicture: filePath },
      { new: true }
    );
    req.session.user.profilePicture = user.profilePicture;
    req.session.success = "Profile picture updated successfully.";
    res.redirect("/profile");
  } catch (error) {
    console.error("Error uploading profile picture:", error);
    res.status(500).render("profile", {
      user: req.session.user,
      error: "Failed to upload profile picture.",
    });
  }
};

const changePassword = async (req, res) => {
  const { currentPassword, password, confirmPassword } = req.body;

  // 1. Check for empty fields
  if (!currentPassword || !password || !confirmPassword) {
    return res.render("profile", {
      user: req.session.user,
      success: null,
      formError: "All fields are required.",
      mismatch: null,
      incorrectPassword: null,
      weakPassword: null,
    });
  }

  // 2. Check password strength
  if (!isStrongPassword(password)) {
    return res.render("profile", {
      user: req.session.user,
      success: null,
      formError: null,
      mismatch: null,
      incorrectPassword: null,
      weakPassword: true,
    });
  }

  // 3. Check for mismatch
  if (password !== confirmPassword) {
    return res.render("profile", {
      user: req.session.user,
      success: null,
      formError: null,
      mismatch: true,
      incorrectPassword: null,
      weakPassword: null,
    });
  }

  // 4. Check if current password is correct
  const user = await User.findById(req.session.user._id);
  const isMatch = await bcrypt.compare(currentPassword, user.password);

  if (!isMatch) {
    return res.render("profile", {
      user: req.session.user,
      success: null,
      formError: null,
      mismatch: null,
      incorrectPassword: true,
      weakPassword: null,
    });
  }
  // 5. All checks passed, update password
  user.password = await bcrypt.hash(password, 10);
  await user.save();

  req.session.success = "Password updated successfully.";
  res.redirect("/profile");
};

const deleteAccount = async (req, res, next) => {
  const { deletePassword, deleteConfirmPassword } = req.body;

  // 1. Check for missing or mismatched passwords
  if (
    !deletePassword ||
    !deleteConfirmPassword ||
    deletePassword !== deleteConfirmPassword
  ) {
    return res.render("profile", {
      user: req.session.user,
      success: null,
      formError: null,
      mismatch: null,
      incorrectPassword: null,
      weakPassword: null,
      deleteError: "Passwords do not match.",
    });
  }

  try {
    // 2. Fetch user and compare password
    const user = await User.findById(req.session.user._id);
    const isMatch = await bcrypt.compare(deletePassword, user.password);

    if (!isMatch) {
      return res.render("profile", {
        user: req.session.user,
        success: null,
        formError: "Incorrect password. Please try again.",
        mismatch: null,
        incorrectPassword: null,
        weakPassword: null,
        deleteError: null,
      });
    }

    // 3. Delete user and logout
    await User.findByIdAndDelete(user._id);

    req.session.destroy((err) => {
      if (err) return next(err);
      res.clearCookie("connect.sid");
      res.cookie("deleteToast", "1", { maxAge: 5000, httpOnly: false });
      res.redirect("/");
    });
  } catch (err) {
    console.error("Delete Account Error:", err.message);
    return res.status(500).render("profile", {
      user: req.session.user,
      success: null,
      formError: "Something went wrong while deleting your account.",
      mismatch: null,
      incorrectPassword: null,
      weakPassword: null,
      deleteError: null,
    });
  }
};

const forgotPassword = async (req, res) => {
  const { email } = req.body;

  try {
    const user = await User.findOne({ email });

    if (!user) {
      return res.render("forgot-password", {
        error: "No account found with that email.",
        success: null,
      });
    }

    if (user.isGoogleUser) {
      return res.render("forgot-password", {
        error:
          "This account was created using Google. Please use Google Sign-In.",
        success: null,
      });
    }

    if (user.resetPasswordExpires && user.resetPasswordExpires > Date.now()) {
      return res.render("forgot-password", {
        error:
          "You've already requested a reset. Please check your email or try again later.",
        success: null,
      });
    }

    const token = crypto.randomBytes(32).toString("hex");
    user.resetPasswordToken = crypto
      .createHash("sha256")
      .update(token)
      .digest("hex");
    user.resetPasswordExpires = Date.now() + 10 * 60 * 1000;

    await user.save();

    const sent = await sendResetEmail(user.email, token);

    if (!sent) {
      return res.render("forgot-password", {
        error: "Failed to send reset email. Please try again.",
        success: null,
      });
    }

    return res.render("forgot-password", {
      success: "A reset link has been sent to your email.",
      error: null,
    });
  } catch (err) {
    console.error("Forgot Password Error:", err.message);
    return res.render("forgot-password", {
      error: "Something went wrong. Please try again.",
      success: null,
    });
  }
};

const showResetPasswordForm = async (req, res) => {
  const token = req.params.token;
  const hashedToken = crypto.createHash("sha256").update(token).digest("hex");

  const user = await User.findOne({
    resetPasswordToken: hashedToken,
    resetPasswordExpires: { $gt: Date.now() },
  });

  if (!user) {
    return res.send("Reset link is invalid or expired");
  }

  res.render("reset-password", { token });
};

const resetPassword = async (req, res) => {
  const { password, confirmPassword } = req.body;
  const token = req.params.token;

  if (password !== confirmPassword) {
    return res.status(400).render("reset-password", {
      token,
      error: "Passwords do not match.",
      success: null,
    });
  }

  if (!isStrongPassword(password)) {
    return res.status(400).render("reset-password", {
      token,
      error:
        "Password must be at least 9 characters and contain a letter and a special character.",
      success: null,
    });
  }

  const hashedToken = crypto.createHash("sha256").update(token).digest("hex");
  const user = await User.findOne({ resetPasswordToken: hashedToken });

  if (!user) {
    return res.status(400).render("reset-used"); // ✅ Used link
  }

  if (!user.resetPasswordExpires || user.resetPasswordExpires < Date.now()) {
    return res.status(400).render("reset-expired"); // ✅ Expired link
  }

  user.password = await bcrypt.hash(password, 10);
  user.resetPasswordToken = undefined;
  user.resetPasswordExpires = undefined;
  await user.save();

  req.session.user = {
    _id: user._id,
    name: user.name,
    email: user.email,
    profilePicture: user.profilePicture || null,
    isGoogleUser: user.isGoogleUser || false,
  };
  
  req.flash("success", "Password updated successfully");
  res.redirect("/");
};

module.exports = {
  registerUser,
  loginUser,
  logoutUser,
  getUserProfile,
  getProfile,
  updateProfile,
  updateProfilePicture,
  changePassword,
  deleteAccount,
  forgotPassword,
  showResetPasswordForm,
  resetPassword,
};
