const User = require("../models/User");
const nodemailer = require("nodemailer");
const crypto = require("crypto");

// Request Password Reset
exports.requestPasswordReset = async (req, res) => {
  try {
    const { email } = req.body;
    const user = await User.findOne({ email });

    if (!user) {
      return res.status(404).json({ message: "User not found" });
    }

    // Generate a token
    const resetToken = crypto.randomBytes(32).toString("hex");

    // Store the token (In a real system, save this to the DB with expiration time)
    user.resetToken = resetToken;
    user.tokenExpiry = Date.now() + 3600000; // 1 hour expiry
    await user.save();

    // Send email (you need to configure an SMTP service like SendGrid or Gmail)
    const transporter = nodemailer.createTransport({
      service: "Gmail",
      auth: {
        user: process.env.SMTP_EMAIL,
        pass: process.env.SMTP_PASSWORD        
      },
    });

    const mailOptions = {
      from: process.env.EMAIL_USER,
      to: email,
      subject: "Password Reset Request",
      html: `
  <h3>Password Reset Request</h3>
  <p>Click the button below to reset your password:</p>
  <a href="http://localhost:5000/reset-password/${resetToken}" 
     style="display:inline-block;padding:12px 24px;background-color:#00c9a7;color:white;border-radius:6px;text-decoration:none;">
     Reset Password
  </a>
  <p>This link will expire in 1 hour.</p>
`,
    };

    await transporter.sendMail(mailOptions);

    res.status(200).json({ message: "Password reset link sent to your email" });
  } catch (error) {
    res.status(500).json({ message: "Error processing request", error });
  }
};

// Reset Password
exports.resetPassword = async (req, res) => {
  try {
    const { token, newPassword } = req.body;
    const user = await User.findOne({
      resetToken: token,
      tokenExpiry: { $gt: Date.now() },
    });

    if (!user) {
      return res.status(400).json({ message: "Invalid or expired token" });
    }

    // Hash new password
    user.password = await bcrypt.hash(newPassword, 10);
    user.resetToken = undefined;
    user.tokenExpiry = undefined;
    await user.save();

    res.status(200).json({ message: "Password reset successfully!" });
  } catch (error) {
    res.status(500).json({ message: "Error resetting password", error });
  }
};
