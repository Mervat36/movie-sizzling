const Search = require("../models/Search");

// Get Scene from History
exports.getSceneHistory = async (req, res) => {
    try {
        const scene = await Search.findOne({ _id: req.params.id });

        if (!scene) {
            return res.status(404).json({ message: "Scene not found" });
        }

        res.status(200).json(scene);
    } catch (error) {
        res.status(500).json({ message: "Error fetching scene", error });
    }
};
