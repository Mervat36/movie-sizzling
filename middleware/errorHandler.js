// 1. Handles all errors and responds with JSON for APIs or renders an error page for browser requests.
const errorHandler = (err, req, res, next) => {
  console.error(err.stack);
  const statusCode = err.statusCode || 500;
  const message = err.message || "Something went wrong";
  // API / AJAX request → return JSON
  if (req.xhr || req.headers.accept.includes("application/json")) {
    return res.status(statusCode).json({
      success: false,
      message,
    });
  }
  // Webpage request → render error.ejs
  res.status(statusCode).render("error", {
    error: {
      status: statusCode,
      message: message,
    },
    theme: req.session?.theme || "light",
  });
};

module.exports = errorHandler;
