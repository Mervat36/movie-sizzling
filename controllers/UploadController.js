// controllers/UploadController.js
const path = require("path");
const fs = require("fs");
const axios = require("axios");
const { exec } = require("child_process");
const supabase = require("../supabaseClient");
const Video = require("../models/Video");
const ShotMetadata = require("../models/ShotData");
const SceneMetadata = require("../models/SceneMetadata");
const SceneResults = require("../models/SceneSearchResult");

exports.handleUpload = async (req, res) => {
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
    const { error } = await supabase.storage
      .from("movies")
      .upload(supaFileName, fileBuffer, {
        contentType: "video/mp4",
        upsert: true,
      });

    if (error) {
      console.error("❌ Supabase upload error:", error);
      return res.status(500).render("error", {
        error: { status: 500, message: "Upload to Supabase failed." },
        theme: req.session.theme || "light",
        friendlyMessage:
          "We couldn't upload your video. Please try again. If the issue persists, contact support.",
      });
    }
    console.log("✅ Supabase video upload complete.");
    fs.unlinkSync(localTempPath);

    req.session.videoTitle = safeTitle;
    console.log("📌 Saved video title to session:", safeTitle);

    const response = await axios({
      method: "get",
      url: publicUrl,
      responseType: "stream",
    });
    const writer = fs.createWriteStream(localDownloadPath);
    response.data.pipe(writer);

    const savedVideo = await Video.create({
      title: movieTitle,
      filename: `dl_${supaFileName}`,
      originalName: req.file.originalname,
      user: req.user._id,
      createdAt: new Date(),
    });

    writer.on("finish", async () => {
      console.log("🟡 Starting shot segmentation...");
      exec(
        `venv\\Scripts\\python.exe AI/shot_segmentation/shot_segmentation.py "${localDownloadPath}" "${safeTitle}"`,
        async (error) => {
          if (error) {
            console.error("❌ Shot model failed:", error.message);
            return res.status(500).render("error", {
              error: { status: 500, message: "Shot segmentation failed." },
              theme: req.session.theme || "light",
              friendlyMessage:
                "We couldn't process the video shots. Try uploading a different video.",
            });
          }

          const jsonPath = path.join("output", `${safeTitle}_shots.json`);
          const shotFolder = path.join("shots", safeTitle);
          if (!fs.existsSync(jsonPath)) {
            return res.status(500).render("error", {
              error: { status: 500, message: "Missing JSON output." },
              theme: req.session.theme || "light",
              friendlyMessage:
                "We couldn't find the generated video data. Please try uploading again.",
            });
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
              const { error: uploadError } = await supabase.storage
                .from("shots")
                .upload(`${safeTitle}/${imageName}`, imageBuffer, {
                  contentType: "image/jpeg",
                  upsert: true,
                });
              if (uploadError) {
                console.error(
                  "❌ Failed to upload shot image:",
                  uploadError.message
                );
                return res
                  .status(500)
                  .send("Shot image upload to Supabase failed.");
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
              return res.status(500).render("error", {
                error: { status: 500, message: "Scene segmentation failed." },
                theme: req.session.theme || "light",
                friendlyMessage:
                  "We couldn’t analyze your video’s scenes. Please try re-uploading the video.",
              });
            }

            console.log("✅ Scene segmentation output:", stdout);

            const scenesJsonPath = path.join(
              "output",
              safeTitle,
              "scenes.json"
            );
            if (!fs.existsSync(scenesJsonPath)) {
              return res.status(500).send("Scenes file missing.");
            }

            const scenesData = JSON.parse(
              fs.readFileSync(scenesJsonPath, "utf-8")
            );

            for (const scene of scenesData.scenes) {
              const localThumbPath = path.join(
                "output",
                safeTitle,
                "scenes",
                path.basename(scene.thumbnail_path)
              );
              if (!fs.existsSync(localThumbPath)) {
                console.warn("Missing scene thumbnail:", localThumbPath);
                continue;
              }

              const buffer = fs.readFileSync(localThumbPath);
              const { error: uploadError } = await supabase.storage
                .from("scene-results")
                .upload(
                  `${safeTitle}/${path.basename(scene.thumbnail_path)}`,
                  buffer,
                  {
                    contentType: "image/jpeg",
                    upsert: true,
                  }
                );

              if (uploadError) {
                console.error(
                  "❌ Scene image upload failed:",
                  uploadError.message
                );
                return res
                  .status(500)
                  .send("Scene image upload to Supabase failed.");
              }

              fs.unlinkSync(localThumbPath);

              const sceneFolder = path.join(
                "output",
                safeTitle,
                "scenes",
                path.basename(scene.thumbnail_path, ".jpg")
              );
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
                    console.warn(
                      "❌ Failed to upload scene shot:",
                      sceneShotError.message
                    );
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

            const localSceneFolder = path.join(
              "output",
              safeTitle,
              "search_scenes"
            );
            fs.mkdirSync(localSceneFolder, { recursive: true });

            console.log("📥 Downloading scene images from Supabase...");
            for (const scene of scenesData.scenes) {
              const baseName = path.basename(scene.thumbnail_path, ".jpg");
              const pattern = new RegExp(
                `^${baseName}_shot\\d+_(start|middle|end)\\.jpg$`
              );
              const { data: files, error: listErr } = await supabase.storage
                .from("scene-results")
                .list(`${safeTitle}`, { limit: 100 });
              if (listErr) {
                console.error(
                  "❌ Failed to list scene files:",
                  listErr.message
                );
                return res.status(500).send("Scene listing failed.");
              }

              const matched = files.filter((f) => pattern.test(f.name));
              for (const file of matched) {
                const { data, error: downloadErr } = await supabase.storage
                  .from("scene-results")
                  .download(`${safeTitle}/${file.name}`);

                if (
                  downloadErr ||
                  !data ||
                  typeof data.arrayBuffer !== "function"
                ) {
                  console.error(
                    "❌ Failed to download or invalid data for:",
                    file.name
                  );
                  continue;
                }

                let buffer;
                try {
                  buffer = Buffer.from(await data.arrayBuffer());
                } catch (e) {
                  console.error(
                    "❌ Buffer conversion failed for:",
                    file.name,
                    e.message
                  );
                  continue;
                }

                fs.writeFileSync(
                  path.join(localSceneFolder, file.name),
                  buffer
                );
              }
            }

            const metadataPath = path.join(
              "output",
              safeTitle,
              "scene_metadata.json"
            );
            const sceneMetadata = await SceneMetadata.findOne({
              title: safeTitle,
            });
            fs.writeFileSync(
              metadataPath,
              JSON.stringify(sceneMetadata, null, 2)
            );

            console.log("🚀 Running scene captioning pipeline...");
            const searchCommand = `set PYTHONPATH=. && venv_search\\Scripts\\python.exe AI/search/search.py "${localSceneFolder}" "${metadataPath}"`;
            exec(searchCommand, async (searchErr, stdout, stderr) => {
              if (
                searchErr &&
                !fs.existsSync(
                  path.join("output", `${safeTitle}_captions.json`)
                )
              ) {
                console.error("❌ Search pipeline failed:", searchErr.message);
                console.error("🔴 STDERR:", stderr);
                return res.status(500).render("error", {
                  error: {
                    status: 500,
                    message: "Scene captioning pipeline failed.",
                  },
                  theme: req.session.theme || "light",
                  friendlyMessage:
                    "Something went wrong while processing your video. Please try again or contact support.",
                });
              }

              console.log("✅ Search captioning completed:\n", stdout);

              console.log("✅ Search captioning completed:\n", stdout);

              console.log("✅ Search captioning completed:\n", stdout);

              // ✅ INSERT THIS BLOCK
              const noScenePath = path.join(
                "output",
                safeTitle,
                "no_scene_found.txt"
              );
              if (fs.existsSync(noScenePath)) {
                console.warn("⚠️ No scene features found.");
                return res.status(404).render("results", {
                  videoPath: null,
                  query: null,
                  message: "No relevant scenes were found for this video.",
                });
              }

              // Continue with regular caption parsing
              try {
                const captionsJsonPath = path.join(
                  process.cwd(),
                  `${safeTitle}_captions.json`
                );

                console.log(
                  "📂 Looking for caption JSON to insert:",
                  captionsJsonPath
                );

                if (!fs.existsSync(captionsJsonPath)) {
                  console.error(
                    "❌ Captions JSON file not found:",
                    captionsJsonPath
                  );
                  return res.status(500).send("Captions file missing.");
                }

                const data = fs.readFileSync(captionsJsonPath, "utf-8").trim();

                if (data.startsWith("<")) {
                  console.error(
                    "❌ Captions file contains unexpected HTML (possibly an error page):\n",
                    data.slice(0, 200)
                  );
                  return res.status(500).render("error", {
                    error: { status: 500, message: "Invalid captions file." },
                    theme: req.session.theme || "light",
                    friendlyMessage:
                      "There was an issue reading your video’s subtitles. Try uploading the video again.",
                  });
                }

                let parsedJson;
                try {
                  parsedJson = JSON.parse(data);
                } catch (parseErr) {
                  console.error(
                    "❌ JSON parsing failed. Content:\n",
                    data.slice(0, 200)
                  );
                  console.error("Error:", parseErr.message);
                  return res.status(500).send("Failed to parse captions file.");
                }

                if (
                  !parsedJson.movie_name ||
                  typeof parsedJson.shots_metadata !== "object"
                ) {
                  console.error(
                    "❌ Invalid caption JSON format. Must contain movie_name and valid shots_metadata."
                  );
                  return res.status(400).send("Invalid caption JSON format.");
                }

                const formattedData = {
                  movie_name: parsedJson.movie_name,
                  shots_metadata: parsedJson.shots_metadata,
                };

                const SceneResults = require("../models/SceneSearchResult");
                await SceneResults.create(formattedData);
                console.log("✅ Captions JSON inserted to MongoDB.");
                return res.redirect(`/search?videoId=${savedVideo._id}`);
              } catch (mongoErr) {
                console.error(
                  "❌ Failed to insert captions JSON to MongoDB:",
                  mongoErr.message
                );
                return res
                  .status(500)
                  .send("Failed to insert captions to Mongo.");
              }
            });
          });
        }
      );
    });

    writer.on("error", (err) => {
      console.error("❌ File write error:", err);
      res.status(500).send("Failed to write downloaded video.");
    });
  } catch (err) {
    console.error("❌ Unexpected error:", err.message);
    res.status(500).send("Server error.");
  }
};
