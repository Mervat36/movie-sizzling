const nodemailer = require("nodemailer");
const ejs = require("ejs");
const path = require("path");

// 1. Sends a password reset email with a token link to the user's email.
async function sendResetEmail(userEmail, token) {
  const templatePath = path.join(
    __dirname,
    "../views/reset-password-email.ejs"
  );
  let emailTemplate;
  587;
  try {
    // Render the email template with the token.
    emailTemplate = await ejs.renderFile(templatePath, { token });
  } catch (err) {
    console.error("❌ Error rendering email template:", err.message);
    return false;
  }
  // 2. Configures the SMTP transporter for sending emails.
  const transporter = nodemailer.createTransport({
    host: "smtp-relay.brevo.com",
    port: 587,
    auth: {
      user: process.env.SMTP_EMAIL,
      pass: process.env.SMTP_PASSWORD,
    },
  });
  const mailOptions = {
    from: "mervat.habib.36@gmail.com",
    to: userEmail,
    subject: "Password Reset Request",
    html: emailTemplate,
  };
  try {
    // 3. Sends the email using the configured transporter.
    await transporter.sendMail(mailOptions);
    console.log("✅ Email sent successfully");
    return true;
  } catch (error) {
    console.error("❌ Email send failed:", error.message);
    return false;
  }
}

module.exports = sendResetEmail;
