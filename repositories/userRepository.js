// repositories/userRepository.js
const User = require('../models/User');

const userRepository = {
  findByEmail: async (email) => await User.findOne({ email }),
  findById: async (id) => await User.findById(id),
  createUser: async (userData) => await User.create(userData),
  updateUser: async (id, data) => await User.findByIdAndUpdate(id, data, { new: true }),
  getAllUsers: async () => await User.find(),
  deleteUser: async (id) => await User.findByIdAndDelete(id)
};

module.exports = userRepository;
