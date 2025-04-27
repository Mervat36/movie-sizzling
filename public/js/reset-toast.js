document.addEventListener("DOMContentLoaded", () => {
  // 1. Handle regular session-based toast.
  if (typeof toast !== "undefined" && toast && toast.message) {
    const toastDiv = document.createElement("div");
    toastDiv.className = `toast ${toast.type}`;
    toastDiv.innerText = toast.message;
    document.body.appendChild(toastDiv);
    setTimeout(() => toastDiv.classList.add("show"), 50);
    setTimeout(() => toastDiv.remove(), 3500);
  }
  // 2. Handle delete account toast from cookie.
  const toastCookie = document.cookie
    .split("; ")
    .find((row) => row.startsWith("deleteToast="));
  if (toastCookie) {
    const toastDiv = document.createElement("div");
    toastDiv.className = "toast success show";
    toastDiv.innerText = "Account deleted successfully.";
    document.body.appendChild(toastDiv);
    setTimeout(() => toastDiv.remove(), 3500);
    document.cookie =
      "deleteToast=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
  }
  // 3. Handle static toast.
  const staticToast = document.getElementById("toast");
  if (staticToast) {
    staticToast.classList.add("show");
    setTimeout(() => staticToast.classList.remove("show"), 4000);
  }
});
