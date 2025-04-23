const nodemailer = require("nodemailer");
const ejs = require("ejs");
const path = require("path");

async function sendResetEmail(userEmail, token) {
  const templatePath = path.join(__dirname, "../views/reset-password-email.ejs");

  let emailTemplate;587
  try {
    emailTemplate = await ejs.renderFile(templatePath, { token });
  } catch (err) {
    console.error("❌ Error rendering email template:", err.message);
    return false;
  }

  const transporter = nodemailer.createTransport({
    host: "smtp-relay.brevo.com",
    port: 587,
    auth: {
      user: process.env.SMTP_EMAIL,
      pass: process.env.SMTP_PASSWORD
    }
  });

  const mailOptions = {
    from: "mervat.habib.36@gmail.com",
    to: userEmail,
    subject: "Password Reset Request",
    html: emailTemplate,
  };

  try {
    await transporter.sendMail(mailOptions);
    console.log("✅ Email sent successfully");
    return true;
  } catch (error) {
    console.error("❌ Email send failed:", error.message);
    return false;
  }
}

module.exports = sendResetEmail;
