# 🎬 Movie Sizzling

Welcome to **Movie Sizzling** — a smart video cataloging platform that helps you search, manage, and explore video content like never before! Whether you're a content creator, teacher, media analyst, or just someone working with lots of video clips, this system is built to make your life easier.

## 🌟 What Does It Do?

- 🎞️ Automatically breaks videos into **shots** and **scenes**
- 🔍 Lets you **search** for specific content using natural language
- 🧠 Uses **AI and deep learning** to recognize objects, actions, and transitions
- 📊 Shows previews, metadata, and matching scenes instantly
- 🔐 Supports secure login, admin control, and social logins via **Google & Facebook**
- 🎯 Works great for education, film, journalism, and video archiving

---

## 🛠 Tech Stack

| Layer         | Tech Used                                 |
|---------------|--------------------------------------------|
| Frontend      | HTML, CSS, JavaScript, EJS                 |
| Backend       | Node.js, Express.js                        |
| Database      | MongoDB + Mongoose                         |
| AI Models     | TensorFlow / PyTorch, TransNetV2, YOLO     |
| Auth          | Passport (Local + OAuth2 for Google/Facebook) |
| Email         | Nodemailer                                 |
| Uploads       | Multer                                     |

---

## 🚀 How to Run It Locally

### 1. Clone the Repo
```bash
git clone https://github.com/mervat36/movie-sizzling.git
cd movie-sizzling
```

### 2. Install the Dependencies
```bash
npm install
```

### 3. Create Your `.env` File
Add a `.env` file in the root folder with:
```env
PORT=5000
MONGO_URI=mongodb://localhost:27017/movie-sizzling
SESSION_SECRET=your_session_secret
EMAIL_USER=your_email@example.com
EMAIL_PASS=your_email_password
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
FACEBOOK_APP_ID=your_facebook_app_id
FACEBOOK_APP_SECRET=your_facebook_secret
```

🛑 **Never commit your `.env` file to GitHub!**

### 4. Start the App
```bash
npm start
# or, if you have nodemon
npx nodemon server.js
```

🖥 Open your browser and go to: [http://localhost:5000](http://localhost:5000)

---

## 🧪 Datasets We Used

- **MovieNet** – for full-scene segmentation
- **ClipShots** – for training shot-detection models

---

## 📂 Project Structure

```
movie-sizzling/
├── public/            # Frontend assets
├── views/             # EJS templates
├── routes/            # Route definitions
├── controllers/       # App logic
├── models/            # Mongoose schemas
├── uploads/           # Video and image uploads
├── middleware/        # Auth and validation
├── .env               # Your local secrets (ignored)
└── server.js
```

---

## 💡 Ideas for the Future

- Real-time scene labeling with WebSockets
- Multi-language subtitle analysis
- Deploying to the cloud with S3 and Docker
- A brand new UI using React or Vue

---

## 👨‍👩‍👧 Team

This project was developed by:
- Mervat Sherif (Team Leader)
- Ahmed Gamal
- Jana Ibrahim
- Sara Mostafa

---

## 📄 License

This project is released under the MIT License — use it, remix it, build on it!

