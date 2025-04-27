# 🎥 Movie Sizzling

Smart Video Segmentation and Search System using AI  
Organize, search, and retrieve specific movie scenes easily!

---

## 📖 Project Overview

Movie Sizzling is a full-stack web application that helps users **upload**, **segment**, **analyze**, and **search** inside videos.  
It uses smart techniques like **scene detection** and **metadata extraction** to allow quick retrieval of specific moments from long videos.

It is designed for content creators, researchers, and media enthusiasts who need **fast access to specific parts of videos**.

---

## ✨ Features

- **User Authentication**
  - Register, Login, Forgot Password, Password Reset
  - Google Login Integration
  - Profile Management (including profile picture updates)
  
- **Video Upload and Management**
  - Upload videos easily
  - Video segmentation into scenes and shots
  - Save metadata and search information
  
- **Smart Search**
  - Search videos by keywords or metadata
  - View results and jump directly to the matching scenes
  
- **Scene History**
  - View previous search history
  - Manage or delete old searches
  
- **Responsive UI**
  - Light and Dark Mode toggle
  - Modern, mobile-friendly design
  - Toast notifications and real-time form validation

---

## 🛠️ Technologies Used

### Backend
- Node.js
- Express.js
- MongoDB + Mongoose
- Supabase (for file storage)

### Frontend
- EJS Templating Engine
- HTML5 / CSS3
- JavaScript (Vanilla)
- Dark Mode Support
- Form Validation (Custom JS)

### Other Tools
- Passport.js (for authentication)
- Multer (for file uploads)
- bcrypt (for password encryption)
- nodemailer (for sending emails)
- dotenv (for environment variables)
- express-session (for session management)

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone <your-repository-link>
cd movie-sizzling
```

### 2. Install dependencies
```bash
npm install
```

### 3. Setup environment variables
Create a `.env` file in the root folder:

```env
MONGO_URI=your_mongodb_connection_string
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_supabase_key
SESSION_SECRET=your_secret_key
EMAIL_USER=your_email@example.com
EMAIL_PASS=your_email_password
BASE_URL=http://localhost:5000
```

### 4. Run the server
```bash
npm run dev
```
Server will start on `http://localhost:5000`

---

## 📂 Project Structure

```
last/
├── controllers/      → Business logic (Upload, Search, User management)
├── middleware/       → Authentication, error handling, file upload config
├── models/           → MongoDB Mongoose schemas
├── public/           → Static assets (CSS, JS, Images)
├── repositories/     → Data handling (example: UserRepository)
├── routes/           → Express route handlers
├── utils/            → Helper functions (like sendResetPasswordEmail)
├── views/            → EJS pages (login, register, profile, upload, search, etc.)
├── server.js         → Entry point of the server
└── README.md         → Project Documentation
```

---

## 🔒 Security Notes

- Passwords are encrypted using `bcrypt`.
- Authentication and authorization are handled with Passport.js.
- Sensitive credentials (database URI, API keys) are stored securely in `.env` (excluded from Git).
- It is recommended to add CSRF protection and rate-limiting middleware in future improvements.

---

## ✍️ Future Improvements

- Add CSRF Protection (`csurf`)
- Add Rate Limiting (`express-rate-limit`) to protect login/signup endpoints
- Add backend validation (`express-validator`)
- Improve animations and UI transitions
- Add support for video previews in search results
- Add admin dashboard for user and video management

---

## 📸 Screenshots

(Add screenshots later: Login Page, Upload Page, Search Results, Profile Page, etc.)

---

# 📢 Credits

- Developed by [Mervat Habib (Team Leader)]
- Ahmed Gamal
- Jana Ibrahim
- Sara Moustafa

---
