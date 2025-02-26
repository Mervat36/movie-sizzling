const Search = require("../models/Search");

// Get search results
exports.getResults = async (req, res) => {
    try {
        const { query } = req.query;
        const results = await Search.find({ sceneDescription: new RegExp(query, "i") });

        if (!results.length) {
            return res.status(404).json({ message: "No results found" });
        }

        res.status(200).json(results);
    } catch (error) {
        res.status(500).json({ message: "Error fetching results", error });
    }
};
