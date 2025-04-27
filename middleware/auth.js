require("dotenv").config();
const passport = require("passport");
const LocalStrategy = require("passport-local").Strategy;
const GoogleStrategy = require("passport-google-oauth20").Strategy;
const FacebookStrategy = require("passport-facebook").Strategy;
const bcrypt = require("bcryptjs");
const User = require("../models/User");

// 1. Local Strategy: Authenticates users using email and password.
passport.use(
  new LocalStrategy(
    { usernameField: "email" },
    async (email, password, done) => {
      try {
        const user = await User.findOne({ email });
        // Check email validation.
        if (!user) {
          return done(null, false, {
            message: "Invalid email. Please try again",
          });
        }
        // Block Google-registered users from logging in manually.
        if (user.isGoogleUser) {
          return done(null, false, {
            message:
              "This email is registered with Google. Please sign in using Google.",
          });
        }
        // Check password.
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

// 2. Authenticates users using Google login.
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
        // Check if user already exists.
        let user = await User.findOne({ email });
        // Add new user.
        if (!user) {
          user = await User.create({
            name: profile.displayName,
            email: email,
            googleId: profile.id,
            provider: "google",
            isGoogleUser: true,
            emailVerified: true,
          });
        }

        return done(null, user);
      } catch (err) {
        return done(err, null);
      }
    }
  )
);

// 3. Stores user ID in session and fetches user on each request.
passport.serializeUser((user, done) => {
  done(null, user._id);
});
passport.deserializeUser(async (id, done) => {
  try {
    const user = await User.findById(id);
    done(null, user);
  } catch (err) {
    done(err, null);
  }
});

// 4. Protects private routes by checking if the user is authenticated.
function ensureAuthenticated(req, res, next) {
  if (req.isAuthenticated?.() || req.session?.user) {
    return next();
  }
  res.redirect("/login");
}

module.exports = {
  ensureAuthenticated,
};
