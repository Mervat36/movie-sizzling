// 📁 middleware/attachUser.js

module.exports = function (req, res, next) {
  if (!req.user && req.session?.user) {
    req.user = req.session.user;
  }
  next();
};
