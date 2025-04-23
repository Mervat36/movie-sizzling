require("dotenv").config();
const passport = require("passport");
const LocalStrategy = require("passport-local").Strategy;
const GoogleStrategy = require("passport-google-oauth20").Strategy;
const FacebookStrategy = require('passport-facebook').Strategy;
const bcrypt = require("bcryptjs");
const User = require("../models/User");

// =============================
// Local Strategy
// =============================
passport.use(
  new LocalStrategy(
    { usernameField: "email" },
    async (email, password, done) => {
      try {
        const user = await User.findOne({ email });

        if (!user) {
          return done(null, false, {
            message: "Invalid email. Please try again"
          });
        }

        // ✅ Block Google-registered users from logging in manually
        if (user.isGoogleUser) {
          return done(null, false, {
            message: "This email is registered with Google. Please sign in using Google.",
          });
        }

        // ✅ Check password
        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch) {
          return done(null, false, {
            message: "Incorrect password. Please try again",
          });
        }

        return done(null, user);
      } catch (err) {
        return done(err);
      }
    }
  )
);

// =============================
// Google OAuth Strategy
// =============================
passport.use(
  new GoogleStrategy(
    {
      clientID: process.env.GOOGLE_CLIENT_ID,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET,
      callbackURL: "/auth/google/callback",
    },
    async (accessToken, refreshToken, profile, done) => {
      try {
        const email = profile.emails[0].value;

        // ✅ Check if user already exists
        let user = await User.findOne({ email });

        if (!user) {
          user = await User.create({
            name: profile.displayName,
            email: email,
            googleId: profile.id,
            provider: "google",
            isGoogleUser: true,
            emailVerified: true
          });
        }

        return done(null, user);
      } catch (err) {
        return done(err, null);
      }
    }
  )
);
// =============================
// Session Handling
// =============================
passport.serializeUser((user, done) => {
  done(null, user._id); // store in session
});

passport.deserializeUser(async (id, done) => {
  try {
    const user = await User.findById(id);
    done(null, user); // adds user to req.user
  } catch (err) {
    done(err, null);
  }
});

// =============================
// Middleware: Protect Routes
// =============================
function ensureAuthenticated(req, res, next) {
  if (req.isAuthenticated?.() || req.session?.user) {
    return next();
  }
  res.redirect("/login");
}

module.exports = {
  ensureAuthenticated,
};
