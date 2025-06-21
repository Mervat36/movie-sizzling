const User = require("../models/User");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const path = require("path");
const crypto = require("crypto");
const sendResetEmail = require("../utils/sendResetPasswordEmail");
const userRepository = require("../repositories/userRepository");

function isStrongPassword(password) {
  return /^(?=.*[A-Za-z])(?=.*[^A-Za-z0-9])(?=.{9,})/.test(password);
}

// 1. Registers a new user (Email or Google account re-registration).
const registerUser = async (req, res) => {
  const { firstName, lastName, email, password, confirmPassword } = req.body;
  try {
    const existingUser = await userRepository.findByEmail(email);
    if (existingUser) {
      if (existingUser.isGoogleUser) {
        // Allow Google account re-registration with password.
        const hashedPassword = await bcrypt.hash(password, 10);
        existingUser.name = `${firstName} ${lastName}`;
        existingUser.password = hashedPassword;
        existingUser.isGoogleUser = false;
        await userRepository.save(existingUser);
        // Update session with all existing user data.
        req.session.user = {
          _id: existingUser._id,
          name: existingUser.name,
          email: existingUser.email,
          profilePicture: existingUser.profilePicture || null,
          isGoogleUser: false,
        };
        return res.redirect("/profile");
      }
      // Normal user already exists.
      return res.status(409).render("register", {
        error: "User already exists.",
        firstName,
        lastName,
        email,
      });
    }
    // Check for mismatch.
    if (password !== confirmPassword) {
      return res.status(400).render("register", {
        error: "Passwords do not match.",
        firstName,
        lastName,
        email,
      });
    }
    // Check password strength.
    if (!isStrongPassword(password)) {
      return res.status(400).render("register", {
        error:
          "Password must be at least 9 characters and contain a letter and a special character.",
        firstName,
        lastName,
        email,
      });
    }
    // Save an Email user in database.
    const hashedPassword = await bcrypt.hash(password, 10);
    const newUser = await userRepository.createUser({
      name: `${firstName} ${lastName}`,
      email,
      password: hashedPassword,
      isGoogleUser: false,
    });
    await newUser.save();
    // Update session with all existing user data.
    req.session.user = {
      _id: newUser._id,
      name: newUser.name,
      email: newUser.email,
      profilePicture: null,
      isGoogleUser: false,
    };
    // All checks passed.
    req.session.toast = {
      type: "success",
      message: "Account created successfully. Welcome!",
    };
    res.redirect("/");
    // Any server error.
  } catch (error) {
    console.error("[Register Error]", error.message);
    return res.status(500).render("register", {
      error: "Server error. Please try again.",
      firstName,
      lastName,
      email,
    });
  }
};

// 2. Login a user (email/password authentication).
const loginUser = async (req, res) => {
  try {
    const { email, password } = req.body;
    // Check for empty fields.
    if (!email || !password) {
      return res.status(400).render("login", {
        error: "Invalid email or password.",
        showPopup: true,
        email,
      });
    }
    const user = await userRepository.findByEmail(email);
    // Check if current email or password is correct.
    if (!user || !(await bcrypt.compare(password, user.password))) {
      return res.status(401).render("login", {
        error: "Invalid email or password.",
        showPopup: true,
        email,
      });
    }
    // Login and open session.
    req.session.user = {
      _id: user._id,
      name: user.name,
      email: user.email,
      profilePicture: user.profilePicture || null,
      isGoogleUser: user.isGoogleUser || false,
      isAdmin: user.isAdmin || false,
    };
    
    // Redirect admin to dashboard, otherwise to homepage
    if (user.isAdmin) {
      res.redirect("/admin");
    } else {
      res.redirect("/");
    }
    // Any server error.
  } catch (error) {
    console.error("[Login Error]", error.message);
    res.status(500).render("login", {
      error: "Server error occurred. Please try again.",
      showPopup: true,
      email,
    });
  }
};

// 3. Logs out the currently logged-in user.
const logoutUser = (req, res) => {
  req.session.destroy(() => {
    res.redirect("/");
  });
};

// 4. Fetch a user's profile information by ID (API JSON).
const getUserProfile = async (req, res) => {
  try {
    const user = await userRepository
      .findById(req.params.id)
      .select("-password");
    if (!user) return res.status(404).json({ message: "User not found" });
    res.status(200).json(user);
  } catch (error) {
    console.error("[Get Profile Error]", error.message);
    res
      .status(500)
      .json({ message: "Error fetching user", error: error.message || error });
  }
};

// 5. Returns the profile page for the logged-in user,
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

// 6. Updates the logged-in user's name and profile picture.
const updateProfile = async (req, res) => {
  try {
    const userId = req.session.user._id;
    const user = await userRepository.findById(userId);
    if (!user) return res.status(404).send("User not found.");
    const { firstName, lastName } = req.body;
    if (firstName && lastName) {
      user.name = `${firstName} ${lastName}`;
    }
    if (req.file) {
      user.profilePicture = `/uploads/profiles/${req.file.filename}`;
    }
    await userRepository.save(user);
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
    // Success update message.
    req.session.success = "Profile updated successfully.";
    res.redirect("/profile");
    // Any server error.
  } catch (err) {
    console.error("Update Error:", err.message);
    res.status(500).render("profile", {
      user: req.session.user,
      error: "Something went wrong while updating your profile.",
    });
  }
};

// 7. Updates only the logged-in user's profile picture.
const updateProfilePicture = async (req, res) => {
  try {
    const filePath = "/uploads/profiles/" + req.file.filename;
    const user = await userRepository.updateProfilePicture(
      req.session.user._id,
      filePath
    );
    // Success update message.
    req.session.user.profilePicture = user.profilePicture;
    req.session.success = "Profile picture updated successfully.";
    res.redirect("/profile");
    // Any server error.
  } catch (error) {
    console.error("[Upload Picture Error]", error.message);
    res.status(500).render("profile", {
      user: req.session.user,
      error: "Failed to upload profile picture.",
    });
  }
};

// 8. Changes the user's password after validating old one.
const changePassword = async (req, res) => {
  const { currentPassword, password, confirmPassword } = req.body;
  // Check for empty fields.
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
  // Check password strength.
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
  // Check for mismatch.
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
  // Check if current password is correct.
  const user = await userRepository.findById(req.session.user._id);
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
  // All checks passed, update password.
  user.password = await bcrypt.hash(password, 10);
  await userRepository.save(user);
  req.session.success = "Password updated successfully.";
  res.redirect("/profile");
};

// 9. Deletes a user's account after password confirmation.
const deleteAccount = async (req, res, next) => {
  const { deletePassword, deleteConfirmPassword } = req.body;
  // Check for missing or mismatched passwords.
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
    // Fetch user and compare password.
    const user = await userRepository.findById(req.session.user._id);
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
    // Delete user and logout.
    await User.findByIdAndDelete(user._id);
    req.session.destroy((err) => {
      if (err) return next(err);
      res.clearCookie("connect.sid");
      res.cookie("deleteToast", "1", { maxAge: 5000, httpOnly: false });
      res.redirect("/");
    });
    // Any server error.
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

// 10. Sends reset email in forgot password.
const forgotPassword = async (req, res) => {
  const { email } = req.body;
  try {
    const user = await userRepository.findByEmail(email);
    // Check if the user's email not found.
    if (!user) {
      return res.render("forgot-password", {
        error: "No account found with that email.",
        success: null,
      });
    }
    // Check if the account is a Google account.
    if (user.isGoogleUser) {
      return res.render("forgot-password", {
        error:
          "This account was created using Google. Please use Google Sign-In.",
        success: null,
      });
    }
    // Check if the reset password is already sent and still valid.
    if (user.resetPasswordExpires && user.resetPasswordExpires > Date.now()) {
      return res.render("forgot-password", {
        error:
          "You've already requested a reset. Please check your email or try again later.",
        success: null,
      });
    }
    // Update the new password.
    const token = crypto.randomBytes(32).toString("hex");
    user.resetPasswordToken = crypto
      .createHash("sha256")
      .update(token)
      .digest("hex");
    user.resetPasswordExpires = Date.now() + 10 * 60 * 1000;
    await userRepository.save(user);
    const sent = await sendResetEmail(user.email, token);
    // Any server error.
    if (!sent) {
      return res.render("forgot-password", {
        error: "Failed to send reset email. Please try again.",
        success: null,
      });
    }
    // All checks passed and the email is sent.
    return res.render("forgot-password", {
      success: "A reset link has been sent to your email.",
      error: null,
    });
    // Any server error.
  } catch (err) {
    console.error("Forgot Password Error:", err.message);
    return res.render("forgot-password", {
      error: "Something went wrong. Please try again.",
      success: null,
    });
  }
};

// 11. Displays the password reset form.
const showResetPasswordForm = async (req, res) => {
  const token = req.params.token;
  const hashedToken = crypto.createHash("sha256").update(token).digest("hex");
  const user = await userRepository.findByResetToken(hashedToken);
  // Check if the link is wrong or invalid.
  if (!user) {
    return res.send("Reset link is invalid or expired");
  }
  // Give a new token to the link.
  res.render("reset-password", { token });
};

// 12. Resets the user's password after verifying the token.
const resetPassword = async (req, res) => {
  const { password, confirmPassword } = req.body;
  const token = req.params.token;
  // Check for mismatch.
  if (password !== confirmPassword) {
    return res.status(400).render("reset-password", {
      token,
      error: "Passwords do not match.",
      success: null,
    });
  }
  // Check for password strength.
  if (!isStrongPassword(password)) {
    return res.status(400).render("reset-password", {
      token,
      error:
        "Password must be at least 9 characters and contain a letter and a special character.",
      success: null,
    });
  }
  const hashedToken = crypto.createHash("sha256").update(token).digest("hex");
  const user = await userRepository.findByResetToken(hashedToken);
  // Reset link is already used once,
  if (!user) {
    return res.status(400).render("reset-used");
  }
  // Reset link is expired.
  if (!user.resetPasswordExpires || user.resetPasswordExpires < Date.now()) {
    return res.status(400).render("reset-expired");
  }
  // Update password successfully.
  user.password = await bcrypt.hash(password, 10);
  user.resetPasswordToken = undefined;
  user.resetPasswordExpires = undefined;
  await userRepository.save(user);
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
