const Search = require("../models/Search");

// Get User Search History
exports.getUserHistory = async (req, res) => {
    try {
        const history = await Search.find({ userId: req.user.id }).sort({ createdAt: -1 });

        if (!history.length) {
            return res.status(404).json({ message: "No history found" });
        }

        res.status(200).json(history);
    } catch (error) {
        res.status(500).json({ message: "Error fetching history", error });
    }
};

// Clear User Search History
exports.clearHistory = async (req, res) => {
    try {
        await Search.deleteMany({ userId: req.user.id });
        res.status(200).json({ message: "Search history cleared successfully!" });
    } catch (error) {
        res.status(500).json({ message: "Error clearing history", error });
    }
};
