document.addEventListener("DOMContentLoaded", () => {
  // ========== 🔍 FILTER FEATURE ==========
  const input = document.getElementById("historySearchInput");

  input?.addEventListener("input", () => {
    const keyword = input.value.toLowerCase();
    const wrappers = document.querySelectorAll(".history-table-wrapper");

    wrappers.forEach(wrapper => {
      const videoCaption = wrapper.querySelector(".caption")?.textContent.toLowerCase();
      const queries = wrapper.querySelectorAll(".history-table");

      let videoMatch = videoCaption.includes(keyword);
      let anyQueryVisible = false;

      queries.forEach(queryTable => {
        const queryText = queryTable.querySelector("th")?.textContent.toLowerCase();
        const match = queryText.includes(keyword);

        queryTable.style.display = match ? "block" : "none";
        if (match) anyQueryVisible = true;
      });

      wrapper.style.display = (videoMatch || anyQueryVisible) ? "block" : "none";
    });

    updateVideoPagination(); // Reset pagination on filter
  });

  // ========== ✅ TOAST FEATURE ==========
  if (typeof toast !== "undefined" && toast && toast.message) {
    const toastDiv = document.createElement("div");
    toastDiv.className = `toast ${toast.type}`;
    toastDiv.innerText = toast.message;
    document.body.appendChild(toastDiv);
    setTimeout(() => toastDiv.classList.add("show"), 50);
    setTimeout(() => toastDiv.remove(), 3500);
  }

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

  // ========== 📄 VIDEO PAGINATION ==========
  const itemsPerPage = 3;
  let currentPage = 1;

  function updateVideoPagination() {
    const wrappers = Array.from(document.querySelectorAll(".history-table-wrapper"))
      .filter(w => w.style.display !== "none");
    const totalPages = Math.ceil(wrappers.length / itemsPerPage);

    wrappers.forEach((wrapper, index) => {
      wrapper.style.display = (index >= (currentPage - 1) * itemsPerPage && index < currentPage * itemsPerPage)
        ? "block"
        : "none";
    });

    renderPaginationControls(totalPages);
  }

  function renderPaginationControls(totalPages) {
    let container = document.getElementById("video-pagination");
    if (!container) {
      container = document.createElement("div");
      container.id = "video-pagination";
      container.style.textAlign = "center";
      container.style.marginTop = "20px";
      document.querySelector(".history-section")?.appendChild(container);
    }
    container.innerHTML = "";

    if (totalPages <= 1) return;

    for (let i = 1; i <= totalPages; i++) {
      const btn = document.createElement("button");
      btn.innerText = i;
      btn.style.margin = "0 5px";
      btn.className = "btn-page";
      if (i === currentPage) btn.style.fontWeight = "bold";
      btn.addEventListener("click", () => {
        currentPage = i;
        updateVideoPagination();
      });
      container.appendChild(btn);
    }
  }

  // ========== 🎞 RESULT PAGINATION ==========
  const resultsPerPage = 3;
  document.querySelectorAll(".history-table").forEach(queryBlock => {
    const results = Array.from(queryBlock.querySelectorAll("tbody tr")).filter(tr => {
      return !tr.querySelector("button.btn-remove")?.closest("form[action^='/queries/delete']");
    });

    const totalResultsPages = Math.ceil(results.length / resultsPerPage);
    if (totalResultsPages <= 1) return;

    let currentResultPage = 1;

    const controlDiv = document.createElement("div");
    controlDiv.className = "results-pagination";
    controlDiv.style.textAlign = "center";
    controlDiv.style.marginTop = "10px";

    for (let i = 1; i <= totalResultsPages; i++) {
      const btn = document.createElement("button");
      btn.innerText = i;
      btn.style.margin = "0 4px";
      btn.className = "btn-page";
      if (i === 1) btn.style.fontWeight = "bold";
      btn.addEventListener("click", () => {
        currentResultPage = i;
        updateResultView();
      });
      controlDiv.appendChild(btn);
    }

    queryBlock.appendChild(controlDiv);

    function updateResultView() {
      results.forEach((row, index) => {
        row.style.display = (index >= (currentResultPage - 1) * resultsPerPage && index < currentResultPage * resultsPerPage)
          ? "table-row"
          : "none";
      });

      controlDiv.querySelectorAll("button").forEach((btn, idx) => {
        btn.style.fontWeight = (idx + 1 === currentResultPage) ? "bold" : "normal";
      });
    }

    updateResultView(); // Initial call
  });

  updateVideoPagination(); // Initial call
});
