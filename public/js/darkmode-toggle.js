document.addEventListener("DOMContentLoaded", () => {
  const toggle = document.getElementById("darkToggle");
  const body = document.body;
  // 1. Check saved preference in localStorage.
  if (localStorage.getItem("darkMode") === "true") {
    toggle.checked = true;
    body.classList.add("dark-mode");
  }
  // 2. Toggle dark mode and update localStorage on change.
  toggle.addEventListener("change", () => {
    body.classList.toggle("dark-mode");
    localStorage.setItem("darkMode", toggle.checked);
  });
});
