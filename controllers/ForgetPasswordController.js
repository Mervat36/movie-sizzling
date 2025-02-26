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
                user: process.env.EMAIL_USER,
                pass: process.env.EMAIL_PASS,
            },
        });

        const mailOptions = {
            from: process.env.EMAIL_USER,
            to: email,
            subject: "Password Reset Request",
            text: `Click the link to reset your password: http://localhost:5000/reset-password/${resetToken}`,
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
        const user = await User.findOne({ resetToken: token, tokenExpiry: { $gt: Date.now() } });

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
