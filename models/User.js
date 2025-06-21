const mongoose = require("mongoose");

const UserSchema = new mongoose.Schema(
  {
    userId: {
      type: Number,
      unique: true,
      sparse: true,
    },
    name: {
      type: String,
      required: true,
      trim: true,
    },
    email: {
      type: String,
      required: true,
      unique: true,
      lowercase: true,
      trim: true,
    },
    password: {
      type: String,
      required: false,
    },
    profilePicture: {
      type: String,
      default: "",
    },
    isGoogleUser: {
      type: Boolean,
      default: false,
    },
    isAdmin: {
      type: Boolean,
      default: false,
    },
    banUntil: { type: Date, default: null },
    resetPasswordToken: String,
    resetPasswordExpires: Date,
  },
  {
    timestamps: true,
  }
);

// 1. Auto-generate userId if not set (e.g. for Google users).
UserSchema.pre("save", async function (next) {
  if (!this.isGoogleUser && !this.password) {
    return next(new Error("Password is required for non-Google users."));
  }
  // Only assign userId if not already set.
  if (!this.userId) {
    const lastUser = await this.constructor
      .findOne()
      .sort({ userId: -1 })
      .exec();
    this.userId = lastUser?.userId ? lastUser.userId + 1 : 1;
  }
  next();
});

module.exports = mongoose.model("User", UserSchema);
