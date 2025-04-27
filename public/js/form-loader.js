document.addEventListener("DOMContentLoaded", () => {
  const forms = document.querySelectorAll("form");
  const spinner = document.getElementById("loading-spinner");
  forms.forEach((form) => {
    form.addEventListener("submit", (event) => {
      // 1. Check if the form is valid before showing spinner.
      if (!form.checkValidity()) {
        return;
      }
      // 2. If form is valid show spinner.
      if (spinner) spinner.style.display = "flex";
    });
  });
});
function showSpinner() {
  const spinner = document.getElementById("loading-spinner");
  if (spinner) {
    spinner.style.display = "flex";
  }
}
