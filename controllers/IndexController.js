exports.homePage = (req, res) => {
    res.sendFile(path.join(__dirname, "../views/index.php"));
};
