document.addEventListener("DOMContentLoaded", () => {
  // ========== 🔍 FILTER FEATURE ==========
  const input = document.getElementById("historySearchInput");

  input?.addEventListener("input", () => {
    const keyword = input.value.toLowerCase();
    const wrappers = document.querySelectorAll(".history-table-wrapper");

    wrappers.forEach((wrapper) => {
      const videoCaption = wrapper
        .querySelector(".caption")
        ?.textContent.toLowerCase();
      const queries = wrapper.querySelectorAll(".history-table, .query-card");

      let videoMatch = videoCaption.includes(keyword);
      let anyQueryVisible = false;

      queries.forEach((queryBlock) => {
        const queryText = queryBlock
          .querySelector("h3")
          ?.textContent.toLowerCase();
        const match = queryText.includes(keyword);

        queryBlock.style.display = match ? "block" : "none";
        if (match) anyQueryVisible = true;
      });

      wrapper.style.display = videoMatch || anyQueryVisible ? "block" : "none";
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
    const wrappers = Array.from(
      document.querySelectorAll(".history-table-wrapper")
    ).filter((w) => w.style.display !== "none");
    const totalPages = Math.ceil(wrappers.length / itemsPerPage);

    wrappers.forEach((wrapper, index) => {
      wrapper.style.display =
        index >= (currentPage - 1) * itemsPerPage &&
        index < currentPage * itemsPerPage
          ? "flex"
          : "none";
    });

    renderPaginationControls(totalPages);
  }

  function setupQueryPagination() {
    document.querySelectorAll(".query-card").forEach((queryCard) => {
      const results = Array.from(queryCard.querySelectorAll(".result-card"));
      if (results.length <= 1) return;

      let current = 0;
      const container = document.createElement("div");
      container.className = "pagination";

      function renderQueryPageButtons() {
        container.innerHTML = "";
        for (let i = 0; i < results.length; i++) {
          const btn = document.createElement("button");
          btn.innerText = i + 1;
          btn.className = "page-btn";
          if (i === current) btn.classList.add("active");
          btn.addEventListener("click", () => {
            results[current].style.display = "none";
            current = i;
            results[current].style.display = "block";
            renderQueryPageButtons();
            results[current].scrollIntoView({
              behavior: "smooth",
              block: "center",
            });
          });

          container.appendChild(btn);
        }
      }

      results.forEach((res, i) => {
        res.style.display = i === 0 ? "block" : "none";
      });

      renderQueryPageButtons();
      queryCard.appendChild(container);
    });
  }

  setupQueryPagination();

  function setupQuerySlider() {
    document.querySelectorAll(".history-table-wrapper").forEach((wrapper) => {
      const cards = wrapper.querySelectorAll(".query-card");
      const navs = wrapper.querySelectorAll(".query-nav");

      if (cards.length <= 1) return;

      navs.forEach((btn) => {
        btn.addEventListener("click", () => {
          const target = parseInt(btn.dataset.idx);
          cards.forEach((card, i) => {
            if (i === target) {
              card.classList.remove("hidden");
            } else {
              card.classList.add("hidden");
            }
          });
          navs.forEach((n) => n.classList.remove("active"));
          btn.classList.add("active");
          cards[target].scrollIntoView({ behavior: "smooth", block: "center" });
        });
      });
    });
  }
  setupQuerySlider();

  function renderPaginationControls(totalPages) {
    let container = document.getElementById("video-pagination");
    if (!container) {
      container = document.createElement("div");
      container.id = "video-pagination";
      container.className = "pagination";
      document.querySelector(".filter-bar")?.after(container);
    }

    container.innerHTML = "";

    if (totalPages <= 1) return;

    const prevBtn = document.createElement("button");
    prevBtn.innerText = "←";
    prevBtn.className = "page-btn";
    prevBtn.disabled = currentPage === 1;
    prevBtn.addEventListener("click", () => {
      if (currentPage > 1) {
        currentPage--;
        updateVideoPagination();
      }
    });
    container.appendChild(prevBtn);

    const nextBtn = document.createElement("button");
    nextBtn.innerText = "→";
    nextBtn.className = "page-btn";
    nextBtn.disabled = currentPage === totalPages;
    nextBtn.addEventListener("click", () => {
      if (currentPage < totalPages) {
        currentPage++;
        updateVideoPagination();
      }
    });
    container.appendChild(nextBtn);
  }

  // ========== 💣 DELETE MODAL ==========
  const modal = document.getElementById("deleteModal");
  const confirmForm = document.getElementById("confirmDeleteForm");
  const cancelBtn = document.getElementById("cancelDelete");
  const modalTitle = document.getElementById("modalTitle");
  const modalText = document.getElementById("modalText");

  document.querySelectorAll(".trigger-delete").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      const deleteUrl = btn.getAttribute("data-delete-url");
      const wrapper = btn.closest(".history-table-wrapper");
      const videoId = wrapper?.getAttribute("data-video-id");

      modalTitle.textContent = "Delete This Video?";
      modalText.textContent =
        "This will permanently delete the video.";

      confirmForm.setAttribute("action", deleteUrl);


      modal.classList.remove("hidden");
    });
  });

  document.querySelectorAll(".trigger-query-delete").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      const queryId = btn.getAttribute("data-query-id");
      const deleteUrl = `/queries/delete/${queryId}`;

      modalTitle.textContent = "Delete This Query?";
      modalText.textContent =
        "This will permanently delete the query only.";
      confirmForm.setAttribute("action", deleteUrl);
      modal.classList.remove("hidden");
    });
  });

  cancelBtn.addEventListener("click", () => {
    modal.classList.add("hidden");
  });

  updateVideoPagination(); // Initial call
});