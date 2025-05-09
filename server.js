// // ===== Start of server.js File 1 =====
// require("dotenv").config();
// const express = require("express");
// const mongoose = require("mongoose");
// const cors = require("cors");
// const morgan = require("morgan");
// const path = require("path");
// const multer = require("multer");
// const { exec } = require("child_process");
// const fs = require("fs");
// const axios = require("axios");
// const session = require("express-session");
// const passport = require("passport");
// const bcrypt = require("bcryptjs");
// const crypto = require("crypto");
// const { ensureAuthenticated } = require("./middleware/auth");
// require("./middleware/auth");
// const { getProfile } = require("./controllers/UserController");

// const User = require("./models/User");
// const supabase = require("./supabaseClient");
// const ShotMetadata = require("./models/ShotData");
// const connectDB = require("./config/db");
// const videoRoutes = require("./routes/VideoRoutes");
// const userRoutes = require("./routes/UserRoutes");
// const searchRoutes = require("./routes/SearchRoutes");
// const uploadDir = path.join(__dirname, "uploads", "profiles");

// // ✅ Ensure upload directory exists
// if (!fs.existsSync(uploadDir)) {
//   fs.mkdirSync(uploadDir, { recursive: true });
// }

// const app = express();
// const PORT = process.env.PORT || 5000;

// // Connect DB
// connectDB();

// // Middleware
// app.use(express.urlencoded({ extended: true }));
// app.use("/uploads", express.static(path.join(__dirname, "uploads")));
// app.use(express.json());
// app.use(cors());
// app.use(morgan("dev"));
// app.use(express.static(path.join(__dirname, "public")));

// // Session and Passport
// app.use(
//   session({
//     secret: process.env.SESSION_SECRET || "your_secret_key",
//     resave: false,
//     saveUninitialized: false,
//   })
// );
// app.use(passport.initialize());
// app.use(passport.session());

// // ✅ Global middleware to pass user to all views
// app.use((req, res, next) => {
//   res.locals.user = req.user || req.session.user || null;
//   res.locals.session = req.session;
//   if (req.session.toast) {
//     res.locals.toast = req.session.toast;
//     delete req.session.toast;
//   }
//   next();
// });

// // View Engine
// app.set("view engine", "ejs");
// app.set("views", path.join(__dirname, "views"));

// // UI Pages
// app.get("/", (req, res) => {
//   const resetParam = req.query.reset;

//   // ❌ If someone adds ?reset=success but no session flag set, block them
//   if (resetParam === "success" && !req.session.resetSuccess) {
//     return res.status(403).render("access-denied"); // or "404" if you prefer
//   }

//   const reset = req.session.resetSuccess ? "success" : null;
//   req.session.resetSuccess = false;

//   res.render("index", {
//     user: req.user || req.session.user,
//     reset: reset
//   });
// });


// app.get("/login", (req, res) => res.render("login"));
// app.get("/register", (req, res) => res.render("register"));
// app.get("/upload", ensureAuthenticated, (req, res) => res.render("upload"));
// app.get("/search", ensureAuthenticated, (req, res) => res.render("search"));
// app.get("/results", ensureAuthenticated, (req, res) => res.render("results"));
// app.get("/history", ensureAuthenticated, (req, res) =>
//   res.render("history", { history: [] })
// );
// app.get("/scene-history", ensureAuthenticated, (req, res) =>
//   res.render("scene-history")
// );
// app.get("/forgot-password", (req, res) => {
//   res.render("forgot-password", {
//     user: req.session.user || null,
//   });
// });

// app.get("/profile", ensureAuthenticated, async (req, res) => {
//   const userId = req.session.user?._id || req.user?._id;
//   if (!userId) return res.redirect("/login");

//   const user = await User.findById(userId);

//   if (req.user) {
//     req.user.name = user.name;
//     req.user.email = user.email;
//     req.user.profilePicture = user.profilePicture;
//     req.user.isGoogleUser = user.isGoogleUser;
//   }

//   const success = req.session.success;
//   delete req.session.success;

//   // ✅ Fix here
//   const [firstName, lastName] = (req.session.user.name || "").split(" ");

//   res.render("profile", {
//     user: req.session.user,
//     firstName,
//     lastName,
//     success,
//     formError: null,
//     mismatch: null,
//     incorrectPassword: null,
//     weakPassword: null
//   });
// });




// // RESET PASSWORD PAGE

// app.post("/set-theme", (req, res) => {
//   const { theme } = req.body;
//   req.session.theme = theme;
//   res.sendStatus(200); // or res.json({ success: true });
// });

// // ✅ RESET PASSWORD SUBMIT & LOGIN
// app.post("/reset-password/:token", async (req, res) => {
//   try {
//     const { password, confirmPassword } = req.body;
//     const token = req.params.token;
//     const hashedToken = crypto.createHash("sha256").update(token).digest("hex");

//     const user = await User.findOne({ resetPasswordToken: hashedToken });

//     if (!user) {
//       return res.status(400).render("reset-used");
//     }

//     if (!user.resetPasswordExpires || user.resetPasswordExpires < Date.now()) {
//       return res.status(400).render("reset-expired");
//     }

//     const passwordRegex = /^(?=.*[A-Za-z])(?=.*[^A-Za-z0-9]).{9,}$/;
//     if (!passwordRegex.test(password)) {
//       return res.render("reset-password", {
//         token,
//         error: "Password must be at least 9 characters and contain a letter and a special character.",
//         success: null,
//       });
//     }

//     if (password !== confirmPassword) {
//       return res.render("reset-password", {
//         token,
//         error: "Passwords do not match.",
//         success: null,
//       });
//     }

//     // ✅ Update password
//     user.password = await bcrypt.hash(password, 10);

//     // ✅ Invalidate the reset token
//     user.resetPasswordToken = undefined;
//     user.resetPasswordExpires = undefined;

//     await user.save();

//     // ✅ Log the user out by destroying the session
//     req.session.resetSuccess = true;

//     // Destroy session securely
//     req.session.regenerate((err) => {
//       if (err) {
//         console.error("Error regenerating session:", err);
//         return res.status(500).send("Error logging out.");
//       }
//       // ✅ Set success flag on new session
//       req.session.resetSuccess = true;
//       return res.redirect("/");
//     });
//   } catch (err) {
//     console.error("Reset error:", err);
//     return res.status(500).send("Server error");
//   }
// });


// // Google OAuth
// app.get(
//   "/auth/google",
//   passport.authenticate("google", { scope: ["profile", "email"] })
// );
// app.get(
//   "/auth/google/callback",
//   passport.authenticate("google", { failureRedirect: "/login" }),
//   async (req, res) => {
//     try {
//       const profile = req.user;
//       let user = await User.findOne({ email: profile.email });

//       if (!user) {
//         user = new User({
//           name: profile.name,
//           email: profile.email,
//           isGoogleUser: true,
//         });
//         await user.save();
//       }

//       req.session.user = {
//         _id: user._id,
//         name: user.name,
//         email: user.email,
//         profilePicture: user.profilePicture || null,
//         isGoogleUser: user.isGoogleUser,
//       };

//       res.redirect("/");
//     } catch (err) {
//       console.error("Google login error:", err.message);
//       res.redirect("/login");
//     }
//   }
// );

// // Multer Config
// const storage = multer.diskStorage({
//   destination: function (req, file, cb) {
//     cb(null, "uploads/");
//   },
//   filename: function (req, file, cb) {
//     const ext = path.extname(file.originalname);
//     const rawTitle = req.body.title || "untitled";
//     const safeTitle = rawTitle.trim().replace(/[^a-z0-9_\-]/gi, "_");
//     cb(null, `${safeTitle}${ext}`);
//   },
// });
// const upload = multer({ storage });



// // APIs
// app.use("/api/videos", videoRoutes);
// app.use("/api/users", userRoutes);
// app.use("/api/search", searchRoutes);
// app.use("/users", userRoutes);
// app.use("/", userRoutes);
// app.use(session({
//   secret: "yourSecretKey",
//   resave: false,
//   saveUninitialized: true,
// }));
// // Start Server
// app.listen(PORT, () => {
//   console.log(`Server running on port ${PORT}`);
// });

// app.post("/upload", upload.single("video"), async (req, res) => {
//   const localTempPath = req.file.path;
//   const movieTitle = req.body.title;
//   const ext = path.extname(req.file.originalname);
//   const safeTitle = movieTitle.trim().replace(/[^a-z0-9_\-]/gi, "_");
//   const supaFileName = `${safeTitle}${ext}`;
//   const publicUrl = `https://tyfttcxihduohajlzmfn.supabase.co/storage/v1/object/public/movies/${supaFileName}`;
//   const localDownloadPath = path.join("uploads", `dl_${supaFileName}`);

//   try {
//     console.log("🟡 Uploading video to Supabase...");
//     const fileBuffer = fs.readFileSync(localTempPath);
//     const { error } = await supabase.storage
//       .from("movies")
//       .upload(supaFileName, fileBuffer, {
//         contentType: "video/mp4",
//         upsert: true,
//       });

//     if (error) {
//       console.error("❌ Supabase upload error:", error);
//       return res.status(500).send("Upload to Supabase failed.");
//     }
//     console.log("✅ Supabase video upload complete.");

//     fs.unlink(localTempPath, (unlinkErr) => {
//       if (unlinkErr)
//         console.warn("⚠️ Failed to delete original upload:", unlinkErr.message);
//     });

//     const response = await axios({
//       method: "get",
//       url: publicUrl,
//       responseType: "stream",
//     });

//     const writer = fs.createWriteStream(localDownloadPath);
//     response.data.pipe(writer);

//     writer.on("finish", async () => {
//       console.log("🟡 Download complete. Starting shot segmentation...");

//       exec(
//         `venv\\Scripts\\python.exe AI/shot_segmentation/shot_segmentation.py "${localDownloadPath}" "${safeTitle}"`,
//         async (error, stdout, stderr) => {
//           console.log("📤 [Model Triggered] Running Python command...");
//           if (error) {
//             console.error(`❌ Python error: ${error.message}`);
//             return res.status(500).send("Model failed.");
//           }

//           console.log("✅ [Shot Segmenter] Finished running shot_segmentation.py");

//           try {
//             const jsonPath = path.join("output", `${safeTitle}_shots.json`);
//             const shotFolder = path.join("shots", safeTitle);
//             const jsonData = JSON.parse(fs.readFileSync(jsonPath, "utf-8"));

//             console.log("🟡 Uploading shots to Supabase...");

//             for (const shot of jsonData.shots) {
//               for (const [key, imagePath] of Object.entries(shot.images)) {
//                 const imageBuffer = fs.readFileSync(imagePath);
//                 const imageName = path.basename(imagePath);

//                 const { error: uploadError } = await supabase.storage
//                   .from("shots")
//                   .upload(`${safeTitle}/${imageName}`, imageBuffer, {
//                     contentType: "image/jpeg",
//                     upsert: true,
//                   });

//                 if (uploadError) {
//                   console.error(`❌ Failed to upload ${imageName}:`, uploadError);
//                 }

//                 fs.unlinkSync(imagePath);
//               }
//             }

//             console.log("✅ [Supabase] All shots uploaded.");

//             const savedMetadata = await ShotMetadata.create(jsonData);
//             console.log("✅ Metadata saved to MongoDB:", savedMetadata._id);

//             fs.rmSync(shotFolder, { recursive: true, force: true });
//             fs.unlink(localDownloadPath, () => {});

//             res.send("✅ Video processed, shots uploaded, metadata saved.");
//           } catch (e) {
//             console.error("❌ Error handling shots or metadata:", e.message);
//             res.status(500).send("Failed to handle shots or metadata.");
//           }
//         }
//       );
//     });

//     writer.on("error", (err) => {
//       console.error("❌ File write error:", err);
//       res.status(500).send("Failed to write downloaded video.");
//     });
//   } catch (err) {
//     console.error("❌ Unexpected error:", err.message);
//     res.status(500).send("Server error.");
//   }
// });

// // 404
// app.use((req, res, next) => {
//   res.status(404).render("error", {
//     error: {
//       status: 404,
//       message: "Page Not Found",
//     },
//     theme: req.session.theme || "light",
//     toast: null // ✅ add this line
//   });
// });

// // 500
// app.use((err, req, res, next) => {
//   console.error(err.stack);
//   res.status(500).render("error", {
//     error: {
//       status: 500,
//       message: "Internal Server Error",
//     },
//     theme: req.session.theme || "light",
//     toast: null // ✅ add this line
//   });
// });


// ===== Start of server.js File 1 =====
require("dotenv").config();
const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const morgan = require("morgan");
const path = require("path");
const multer = require("multer");
const { exec } = require("child_process");
const fs = require("fs");
const axios = require("axios");
const session = require("express-session");
const passport = require("passport");
const bcrypt = require("bcryptjs");
const crypto = require("crypto");
const { ensureAuthenticated } = require("./middleware/auth");
require("./middleware/auth");
require('events').EventEmitter.defaultMaxListeners = 20;

const { getProfile } = require("./controllers/UserController");

const User = require("./models/User");
const supabase = require("./supabaseClient");
const ShotMetadata = require("./models/ShotData");

const UserQuery = require("./models/UserQuery");

const SceneMetadata = require("./models/SceneMetadata");
const SceneResults = require("./models/SceneSearchResult");


const connectDB = require("./config/db");
const videoRoutes = require("./routes/VideoRoutes");
const userRoutes = require("./routes/UserRoutes");
const searchRoutes = require("./routes/SearchRoutes");
const uploadDir = path.join(__dirname, "uploads", "profiles");

// ✅ Ensure upload directory exists
if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir, { recursive: true });
}

const app = express();
const PORT = process.env.PORT || 5000;

// Connect DB
connectDB();

// Middleware
app.use(express.urlencoded({ extended: true }));
app.use("/uploads", express.static(path.join(__dirname, "uploads")));
app.use(express.json());
app.use(cors());
app.use(morgan("dev"));
app.use(express.static(path.join(__dirname, "public")));

// Session and Passport
app.use(
  session({
    secret: process.env.SESSION_SECRET || "your_secret_key",
    resave: false,
    saveUninitialized: false,
  })
);
app.use(passport.initialize());
app.use(passport.session());

// ✅ Global middleware to pass user to all views
app.use((req, res, next) => {
  res.locals.user = req.user || req.session.user || null;
  res.locals.session = req.session;
  if (req.session.toast) {
    res.locals.toast = req.session.toast;
    delete req.session.toast;
  }
  next();
});

// View Engine
app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "views"));

// UI Pages
app.get("/", (req, res) => {
  const resetParam = req.query.reset;

  // ❌ If someone adds ?reset=success but no session flag set, block them
  if (resetParam === "success" && !req.session.resetSuccess) {
    return res.status(403).render("access-denied"); // or "404" if you prefer
  }

  const reset = req.session.resetSuccess ? "success" : null;
  req.session.resetSuccess = false;

  res.render("index", {
    user: req.user || req.session.user,
    reset: reset
  });
});


app.get("/login", (req, res) => res.render("login"));
app.get("/register", (req, res) => res.render("register"));
app.get("/upload", ensureAuthenticated, (req, res) => res.render("upload"));
app.get("/search", ensureAuthenticated, (req, res) => res.render("search"));
app.get("/results", ensureAuthenticated, (req, res) => res.render("results"));
app.get("/history", ensureAuthenticated, (req, res) =>
  res.render("history", { history: [] })
);
app.get("/scene-history", ensureAuthenticated, (req, res) =>
  res.render("scene-history")
);
app.get("/forgot-password", (req, res) => {
  res.render("forgot-password", {
    user: req.session.user || null,
  });
});

app.get("/profile", ensureAuthenticated, async (req, res) => {
  const userId = req.session.user?._id || req.user?._id;
  if (!userId) return res.redirect("/login");

  const user = await User.findById(userId);

  if (req.user) {
    req.user.name = user.name;
    req.user.email = user.email;
    req.user.profilePicture = user.profilePicture;
    req.user.isGoogleUser = user.isGoogleUser;
  }

  const success = req.session.success;
  delete req.session.success;

  // ✅ Fix here
  const [firstName, lastName] = (req.session.user.name || "").split(" ");

  res.render("profile", {
    user: req.session.user,
    firstName,
    lastName,
    success,
    formError: null,
    mismatch: null,
    incorrectPassword: null,
    weakPassword: null
  });
});




// RESET PASSWORD PAGE

app.post("/set-theme", (req, res) => {
  const { theme } = req.body;
  req.session.theme = theme;
  res.sendStatus(200); // or res.json({ success: true });
});

// ✅ RESET PASSWORD SUBMIT & LOGIN
app.post("/reset-password/:token", async (req, res) => {
  try {
    const { password, confirmPassword } = req.body;
    const token = req.params.token;
    const hashedToken = crypto.createHash("sha256").update(token).digest("hex");

    const user = await User.findOne({ resetPasswordToken: hashedToken });

    if (!user) {
      return res.status(400).render("reset-used");
    }

    if (!user.resetPasswordExpires || user.resetPasswordExpires < Date.now()) {
      return res.status(400).render("reset-expired");
    }

    const passwordRegex = /^(?=.*[A-Za-z])(?=.*[^A-Za-z0-9]).{9,}$/;
    if (!passwordRegex.test(password)) {
      return res.render("reset-password", {
        token,
        error: "Password must be at least 9 characters and contain a letter and a special character.",
        success: null,
      });
    }

    if (password !== confirmPassword) {
      return res.render("reset-password", {
        token,
        error: "Passwords do not match.",
        success: null,
      });
    }

    // ✅ Update password
    user.password = await bcrypt.hash(password, 10);

    // ✅ Invalidate the reset token
    user.resetPasswordToken = undefined;
    user.resetPasswordExpires = undefined;

    await user.save();

    // ✅ Log the user out by destroying the session
    req.session.resetSuccess = true;

    // Destroy session securely
    req.session.regenerate((err) => {
      if (err) {
        console.error("Error regenerating session:", err);
        return res.status(500).send("Error logging out.");
      }
      // ✅ Set success flag on new session
      req.session.resetSuccess = true;
      return res.redirect("/");
    });
  } catch (err) {
    console.error("Reset error:", err);
    return res.status(500).send("Server error");
  }
});


// Google OAuth
app.get(
  "/auth/google",
  passport.authenticate("google", { scope: ["profile", "email"] })
);
app.get(
  "/auth/google/callback",
  passport.authenticate("google", { failureRedirect: "/login" }),
  async (req, res) => {
    try {
      const profile = req.user;
      let user = await User.findOne({ email: profile.email });

      if (!user) {
        user = new User({
          name: profile.name,
          email: profile.email,
          isGoogleUser: true,
        });
        await user.save();
      }

      req.session.user = {
        _id: user._id,
        name: user.name,
        email: user.email,
        profilePicture: user.profilePicture || null,
        isGoogleUser: user.isGoogleUser,
      };

      res.redirect("/");
    } catch (err) {
      console.error("Google login error:", err.message);
      res.redirect("/login");
    }
  }
);

// Multer Config
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, "uploads/");
  },
  filename: function (req, file, cb) {
    const ext = path.extname(file.originalname);
    const rawTitle = req.body.title || "untitled";
    const safeTitle = rawTitle.trim().replace(/[^a-z0-9_\-]/gi, "_");
    cb(null, `${safeTitle}${ext}`);
  },
});
const upload = multer({ storage });



// APIs
app.use("/api/videos", videoRoutes);
app.use("/api/users", userRoutes);
app.use("/api/search", searchRoutes);
app.use("/users", userRoutes);
app.use("/", userRoutes);
app.use(session({
  secret: "yourSecretKey",
  resave: false,
  saveUninitialized: true,
}));
// Start Server
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});



app.post("/search", ensureAuthenticated, async (req, res) => {
  const { query } = req.body;
  const userId = req.user?._id || req.session.user?._id;

  // ✅ Get videoTitle from session
  const safeTitle = req.session.videoTitle;

  if (!safeTitle) {
    return res.status(400).send("No video linked to this search.");
  }

  try {
    const newQuery = new UserQuery({ userId, query });
    await newQuery.save();

    const captionsPath = path.join(__dirname, `${safeTitle}_captions.json`);
    const videoPath = path.join("uploads", `dl_${safeTitle}.mp4`);

    if (!fs.existsSync(captionsPath) || !fs.existsSync(videoPath)) {
      return res.status(400).send("Captions or video not found.");
    }

    const pyCommand = `set PYTHONPATH=. && venv_search\\Scripts\\python.exe AI/search/search_user_query.py "${captionsPath}" "${query}" "${videoPath}"`;

    exec(pyCommand, async (error, stdout, stderr) => {
      if (error) {
        console.error("❌ Python search failed:", error.message);
        console.error("🔴 STDERR:", stderr);
        return res.status(500).send("Search failed.");
      }

      console.log("✅ Search finished:", stdout);
      res.redirect("/search-results");
    });

  } catch (err) {
    console.error("❌ Error saving query or executing search:", err.message);
    res.status(500).send("Internal Server Error");
  }
});




app.post("/upload", upload.single("video"), async (req, res) => {
  const localTempPath = req.file.path;
  const movieTitle = req.body.title;
  const ext = path.extname(req.file.originalname);
  const safeTitle = movieTitle.trim().replace(/[^a-z0-9_\-]/gi, "_");
  const supaFileName = `${safeTitle}${ext}`;
  const publicUrl = `https://tyfttcxihduohajlzmfn.supabase.co/storage/v1/object/public/movies/${supaFileName}`;
  const localDownloadPath = path.join("uploads", `dl_${supaFileName}`);

  try {
    console.log("🟡 Uploading video to Supabase...");
    const fileBuffer = fs.readFileSync(localTempPath);
    const { error } = await supabase.storage.from("movies").upload(supaFileName, fileBuffer, {
      contentType: "video/mp4",
      upsert: true,
    });

    if (error) {
      console.error("❌ Supabase upload error:", error);
      return res.status(500).send("Upload to Supabase failed.");
    }
    console.log("✅ Supabase video upload complete.");
    fs.unlinkSync(localTempPath);

    req.session.videoTitle = safeTitle;
    console.log("📌 Saved video title to session:", safeTitle);

    const response = await axios({ method: "get", url: publicUrl, responseType: "stream" });
    const writer = fs.createWriteStream(localDownloadPath);
    response.data.pipe(writer);

    
    writer.on("finish", async () => {
      console.log("🟡 Starting shot segmentation...");
      exec(`venv\\Scripts\\python.exe AI/shot_segmentation/shot_segmentation.py "${localDownloadPath}" "${safeTitle}"`, async (error) => {
        if (error) {
          console.error("❌ Shot model failed:", error.message);
          return res.status(500).send("Shot model failed.");
        }

        const jsonPath = path.join("output", `${safeTitle}_shots.json`);
        const shotFolder = path.join("shots", safeTitle);
        if (!fs.existsSync(jsonPath)) {
          return res.status(500).send("Shot JSON file missing.");
        }

        let jsonData;
        try {
          jsonData = JSON.parse(fs.readFileSync(jsonPath, "utf-8"));
        } catch (err) {
          return res.status(500).send("Invalid shot JSON format.");
        }

        console.log("🟡 Uploading shots to Supabase...");
        const allShotImagePaths = [];
        for (const shot of jsonData.shots) {
          for (const [key, imagePath] of Object.entries(shot.images)) {
            const imageBuffer = fs.readFileSync(imagePath);
            const imageName = path.basename(imagePath);
            const { error: uploadError } = await supabase.storage.from("shots").upload(`${safeTitle}/${imageName}`, imageBuffer, {
              contentType: "image/jpeg",
              upsert: true,
            });
            if (uploadError) {
              console.error("❌ Failed to upload shot image:", uploadError.message);
              return res.status(500).send("Shot image upload to Supabase failed.");
            }
            allShotImagePaths.push(imagePath);
          }
        }

        await ShotMetadata.create(jsonData);
        // fs.unlinkSync(localDownloadPath);

        console.log("🟢 Starting scene segmentation...");
        const sceneCommand = `set PYTHONPATH=.&& venv_scene_class\\Scripts\\python.exe AI/Scene/model/scene_segmentation.py "${safeTitle}" "output/${safeTitle}_shots.json" "shots/${safeTitle}"`;

        exec(sceneCommand, async (sceneErr, stdout, stderr) => {
          if (sceneErr) {
            console.error("❌ Scene segmentation failed:", sceneErr.message);
            console.error("🔴 STDERR:", stderr);
            return res.status(500).send("Scene segmentation failed.");
          }

          console.log("✅ Scene segmentation output:", stdout);

          const scenesJsonPath = path.join("output", safeTitle, "scenes.json");
          if (!fs.existsSync(scenesJsonPath)) {
            return res.status(500).send("Scenes file missing.");
          }

          const scenesData = JSON.parse(fs.readFileSync(scenesJsonPath, "utf-8"));

          for (const scene of scenesData.scenes) {
            const localThumbPath = path.join("output", safeTitle, "scenes", path.basename(scene.thumbnail_path));
            if (!fs.existsSync(localThumbPath)) {
              console.warn("Missing scene thumbnail:", localThumbPath);
              continue;
            }

            const buffer = fs.readFileSync(localThumbPath);
            const { error: uploadError } = await supabase.storage
              .from("scene-results")
              .upload(`${safeTitle}/${path.basename(scene.thumbnail_path)}`, buffer, {
                contentType: "image/jpeg",
                upsert: true,
              });

            if (uploadError) {
              console.error("❌ Scene image upload failed:", uploadError.message);
              return res.status(500).send("Scene image upload to Supabase failed.");
            }

            fs.unlinkSync(localThumbPath);

            const sceneFolder = path.join("output", safeTitle, "scenes", path.basename(scene.thumbnail_path, ".jpg"));
            if (fs.existsSync(sceneFolder)) {
              const sceneFiles = fs.readdirSync(sceneFolder);
              for (const file of sceneFiles) {
                const filePath = path.join(sceneFolder, file);
                const fileBuffer = fs.readFileSync(filePath);

                const { error: sceneShotError } = await supabase.storage
                  .from("scene-results")
                  .upload(`${safeTitle}/${file}`, fileBuffer, {
                    contentType: "image/jpeg",
                    upsert: true,
                  });

                if (sceneShotError) {
                  console.warn("❌ Failed to upload scene shot:", sceneShotError.message);
                }

                fs.unlinkSync(filePath);
              }

              fs.rmdirSync(sceneFolder);
            }
          }

          await SceneMetadata.create({
            title: safeTitle,
            scenes: scenesData.scenes,
          });

          for (const imgPath of allShotImagePaths) {
            if (fs.existsSync(imgPath)) fs.unlinkSync(imgPath);
          }
          fs.rmSync(shotFolder, { recursive: true, force: true });

          const localSceneFolder = path.join("output", safeTitle, "search_scenes");
          fs.mkdirSync(localSceneFolder, { recursive: true });

          console.log("📥 Downloading scene images from Supabase...");
          for (const scene of scenesData.scenes) {
            const baseName = path.basename(scene.thumbnail_path, ".jpg");
            const pattern = new RegExp(`^${baseName}_shot\\d+_(start|middle|end)\\.jpg$`);
            const { data: files, error: listErr } = await supabase.storage.from("scene-results").list(`${safeTitle}`, { limit: 100 });
            if (listErr) {
              console.error("❌ Failed to list scene files:", listErr.message);
              return res.status(500).send("Scene listing failed.");
            }

            const matched = files.filter(f => pattern.test(f.name));
            for (const file of matched) {
              const { data, error: downloadErr } = await supabase.storage.from("scene-results").download(`${safeTitle}/${file.name}`);
              if (downloadErr) {
                console.error("❌ Failed to download:", file.name);
                continue;
              }
              const buffer = Buffer.from(await data.arrayBuffer());
              fs.writeFileSync(path.join(localSceneFolder, file.name), buffer);
            }
          }

          const metadataPath = path.join("output", safeTitle, "scene_metadata.json");
          const sceneMetadata = await SceneMetadata.findOne({ title: safeTitle });
          fs.writeFileSync(metadataPath, JSON.stringify(sceneMetadata, null, 2));

          console.log("🚀 Running scene captioning pipeline...");
          const searchCommand = `set PYTHONPATH=. && venv_search\\Scripts\\python.exe AI/search/search.py "${localSceneFolder}" "${metadataPath}"`;
          exec(searchCommand, async (searchErr, stdout, stderr) => {
            if (searchErr) {
              console.error("❌ Search pipeline failed:", searchErr.message);
              console.error("🔴 STDERR:", stderr);
              return res.status(500).send("Scene captioning pipeline failed.");
            }

            console.log("✅ Search captioning completed:\n", stdout);

            try {
              const captionsJsonPath = path.join(__dirname, `${safeTitle}_captions.json`);
              console.log("📂 Looking for caption JSON to insert:", captionsJsonPath);

              if (!fs.existsSync(captionsJsonPath)) {
                console.error("❌ Captions JSON file not found:", captionsJsonPath);
                return res.status(500).send("Captions file missing.");
              }

              const data = fs.readFileSync(captionsJsonPath, "utf-8");
              if (data.trim().startsWith("<")) {
                console.error("❌ Captions file contains unexpected HTML instead of JSON.");
                return res.status(500).send("Invalid captions file content.");
              }

              let parsedJson;
              try {
                parsedJson = JSON.parse(data);
              } catch (parseErr) {
                console.error("❌ Failed to parse captions JSON:", parseErr.message);
                return res.status(500).send("Failed to parse captions file.");
              }

              if (!parsedJson.movie_name || typeof parsedJson.shots_metadata !== 'object') {
                console.error("❌ Invalid caption JSON format. Must contain movie_name and valid shots_metadata.");
                return res.status(400).send("Invalid caption JSON format.");
              }

              const formattedData = {
                movie_name: parsedJson.movie_name,
                shots_metadata: parsedJson.shots_metadata
              };

              const SceneResults = require("./models/SceneSearchResult");
              await SceneResults.create(formattedData);
              console.log("✅ Captions JSON inserted to MongoDB.");
              res.render("search", {
                uploadedVideoTitle: safeTitle

              });
            } catch (mongoErr) {
              console.error("❌ Failed to insert captions JSON to MongoDB:", mongoErr.message);
              return res.status(500).send("Failed to insert captions to Mongo.");
            }
          });
        });
      });
    });

    writer.on("error", (err) => {
      console.error("❌ File write error:", err);
      res.status(500).send("Failed to write downloaded video.");
    });
  } catch (err) {
    console.error("❌ Unexpected error:", err.message);
    res.status(500).send("Server error.");
  }
});