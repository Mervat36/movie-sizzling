const mongoose = require("mongoose");

const SearchSchema = new mongoose.Schema({
    query: String,
    results: Array,
}, { timestamps: true });

module.exports = mongoose.model("Search", SearchSchema);
