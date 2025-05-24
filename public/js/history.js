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

    let visibleCount = 0;
    wrappers.forEach((wrapper) => {
      const isVisible = wrapper.style.display !== "none";
      if (isVisible) {
        visibleCount++;
        const start = (currentPage - 1) * itemsPerPage + 1;
        const end = currentPage * itemsPerPage;
        wrapper.style.display =
          visibleCount >= start && visibleCount <= end ? "flex" : "none";
      }
    });

    renderPaginationControls(totalPages);
  }

  function setupQueryPagination() {
    document.querySelectorAll(".query-card").forEach((queryCard) => {
      const results = Array.from(queryCard.querySelectorAll(".result-card"));
      if (results.length <= 1) return;

      // ✅ Remove any existing pagination container before adding a new one
      const existingPagination = queryCard.querySelector(".pagination");
      if (existingPagination) existingPagination.remove();

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

  cancelBtn.addEventListener("click", () => {
    modal.classList.add("hidden");
  });

  updateVideoPagination(); // Initial call
  document
    .querySelectorAll(".trigger-delete, .trigger-query-delete")
    .forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.preventDefault();

        const modal = document.getElementById("deleteModal");
        const modalTitle = document.getElementById("modalTitle");
        const modalText = document.getElementById("modalText");
        const confirmForm = document.getElementById("confirmDeleteForm");

        let deleteUrl = "";
        let deleteTarget = null;
        let deleteType = "";

        if (btn.classList.contains("trigger-query-delete")) {
          deleteUrl = `/queries/delete/${btn.dataset.queryId}`;
          deleteTarget = btn.closest(".query-card");
          deleteType = "query";
          modalTitle.textContent = "Delete Query?";
          modalText.textContent =
            "This will permanently delete the query and its results.";
        } else if (btn.closest(".result-card")) {
          deleteUrl = btn.dataset.deleteUrl;
          deleteTarget = btn.closest(".result-card");
          deleteType = "result";
          modalTitle.textContent = "Delete Result?";
          modalText.textContent = "This will permanently delete the result.";
        } else {
          deleteUrl = btn.dataset.deleteUrl;
          deleteTarget = btn.closest(".history-table-wrapper");
          deleteType = "video";
          modalTitle.textContent = "Delete Video?";
          modalText.textContent =
            "This will permanently delete the video and all related data.";
        }

        modal.classList.remove("hidden");

        confirmForm.onsubmit = async function (ev) {
          ev.preventDefault();
          try {
            const res = await fetch(deleteUrl, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                Accept: "application/json",
              },
            });

            let json;
            try {
              json = await res.json();
            } catch (err) {
              console.warn("Failed to parse JSON response", err);
              showToast(
                `An error occurred while deleting the ${deleteType}.`,
                "error"
              );
              modal.classList.add("hidden");
              return;
            }

            if (!res.ok || !json.success) {
              showToast(`Failed to delete ${deleteType}.`, "error");
              modal.classList.add("hidden");
              return;
            }

            if (deleteType === "result") {
              const resultCard = deleteTarget;
              const queryCard = resultCard.closest(".query-card");
              const panel = queryCard.closest(".query-panel");
              const panelId = panel.id;
              const tab = document.querySelector(
                `.query-tab[data-tab="${panelId}"]`
              );
              const queryList = panel.closest(".query-list");

              resultCard.remove();
              const remainingResults =
                queryCard.querySelectorAll(".result-card");

              if (remainingResults.length === 0) {
                // Now make an additional call to delete the entire query on the backend
                const queryDeleteBtn = queryCard.querySelector(
                  ".trigger-query-delete"
                );
                const queryId = queryDeleteBtn?.dataset.queryId;

                if (queryId) {
                  const queryDeleteRes = await fetch(
                    `/queries/delete/${queryId}`,
                    {
                      method: "POST",
                      headers: {
                        "Content-Type": "application/json",
                        Accept: "application/json",
                      },
                    }
                  );

                  let deleteSuccess = false;
                  try {
                    const json = await queryDeleteRes.json();
                    deleteSuccess = json.success;
                  } catch {
                    deleteSuccess = false;
                  }

                  if (!queryDeleteRes.ok || !deleteSuccess) {
                    showToast(
                      "An error occurred while deleting the query.",
                      "error"
                    );
                    modal.classList.add("hidden");
                    return;
                  }

                  queryCard.remove();
                  panel.remove();
                  tab.remove();

                  const remainingPanels =
                    queryList.querySelectorAll(".query-panel");
                  if (remainingPanels.length === 0) {
                    queryList.innerHTML = `<p>You haven't searched any queries for this video yet.</p>`;
                  } else {
                    const nextTab = document.querySelector(".query-tab");
                    const nextPanelId = nextTab?.dataset.tab;
                    if (nextPanelId) {
                      document
                        .getElementById(nextPanelId)
                        ?.classList.add("active");
                      nextTab?.classList.add("active");
                    }
                  }

                  updateVideoPagination();
                  showToast("Result deleted successfully!");
                }
              } else {
                // More results left — reset pagination
                remainingResults.forEach((res, i) => {
                  res.style.display = i === 0 ? "block" : "none";
                });

                const pagination = queryCard.querySelector(".pagination");
                if (pagination) pagination.remove();

                setupQueryPagination();
                updateVideoPagination();
                showToast("Result deleted successfully!");
              }
            } else if (deleteType === "query") {
              const queryCard = deleteTarget;
              const panel = queryCard.closest(".query-panel");
              const panelId = panel.id;
              const tab = document.querySelector(
                `.query-tab[data-tab="${panelId}"]`
              );
              const queryList = panel.closest(".query-list");

              queryCard.remove();
              panel.remove();
              tab.remove();

              const remainingPanels =
                queryList.querySelectorAll(".query-panel");
              if (remainingPanels.length === 0) {
                queryList.innerHTML = `<p>You haven't searched any queries for this video yet.</p>`;
              } else {
                const tabsInGroup = queryList.querySelectorAll(".query-tab");
                const panelsInGroup =
                  queryList.querySelectorAll(".query-panel");

                // Deactivate all remaining tabs and panels
                tabsInGroup.forEach((t) => t.classList.remove("active"));
                panelsInGroup.forEach((p) => p.classList.remove("active"));

                // Activate first remaining
                const firstTab = tabsInGroup[0];
                const firstPanelId = firstTab.dataset.tab;
                const firstPanel = document.getElementById(firstPanelId);

                if (firstTab && firstPanel) {
                  firstTab.classList.add("active");
                  firstPanel.classList.add("active");
                  firstPanel.scrollIntoView({
                    behavior: "smooth",
                    block: "center",
                  });
                }
              }

              updateVideoPagination();
              showToast("Query deleted successfully!");
            } else if (deleteType === "video") {
              deleteTarget.remove();

              // Recalculate visible videos and pagination
              const allVideos = document.querySelectorAll(
                ".history-table-wrapper"
              );
              const visibleVideos = Array.from(allVideos).filter(
                (v) => v.style.display !== "none"
              );

              const wrappersPerPage = itemsPerPage;
              const remainingOnPage = visibleVideos.length;

              // Determine if this was the last video on the current page
              const totalVideos = document.querySelectorAll(
                ".history-table-wrapper"
              ).length;
              const totalPages = Math.ceil(totalVideos / wrappersPerPage);

              if (remainingOnPage === 0 && totalPages > 1) {
                // We're on a now-empty page, go back to the first page
                currentPage = 1;
              }

              updateVideoPagination();

              // Check again after pagination update
              const finalVisible = document.querySelectorAll(
                ".history-table-wrapper"
              );
              if (finalVisible.length === 0) {
                document.querySelector(".history-section").innerHTML = `
      <div class="empty-state">
        <img src="/images/no-videos.png" alt="No videos" class="empty-illustration" />
        <h3 class="empty-title">You haven't uploaded any videos yet</h3>
        <p class="empty-subtext">
          Start by uploading your first video to begin searching and saving results.
        </p>
        <a href="/upload" class="btn-primary">Upload a Video</a>
      </div>`;
              }

              showToast("Video deleted successfully!");
            } else {
              // Handle other types like query and video as you already do
              deleteTarget.remove();
              updateVideoPagination();
              showToast(
                `${
                  deleteType.charAt(0).toUpperCase() + deleteType.slice(1)
                } deleted successfully!`
              );
            }
          } catch (err) {
            showToast(
              `An error occurred while deleting the ${deleteType}.`,
              "error"
            );
          }

          modal.classList.add("hidden");
        };
      });
    });

  // ========== ✅ QUERY TABS SWITCHING ==========
  function setupQueryTabs() {
    const tabs = document.querySelectorAll(".query-tab");
    const panels = document.querySelectorAll(".query-panel");

    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const targetId = tab.getAttribute("data-tab");

        // Deactivate all tabs and panels
        tabs.forEach((t) => t.classList.remove("active"));
        panels.forEach((p) => p.classList.remove("active"));

        // Activate the selected tab and panel
        tab.classList.add("active");
        const targetPanel = document.getElementById(targetId);
        if (targetPanel) {
          targetPanel.classList.add("active");
        }
      });
    });
  }
  // ========== 🔍 SEARCH FILTER FOR VIDEOS AND QUERIES ==========
  const searchInput = document.getElementById("historySearchInput");
  searchInput?.addEventListener("input", function () {
    const term = this.value.toLowerCase();
    document.querySelectorAll(".history-table-wrapper").forEach((wrapper) => {
      const caption =
        wrapper.querySelector(".caption")?.textContent.toLowerCase() || "";
      const queries = [...wrapper.querySelectorAll(".query-tab")].map((btn) =>
        btn.textContent.toLowerCase()
      );
      const matches =
        caption.includes(term) || queries.some((q) => q.includes(term));
      wrapper.style.display = matches ? "flex" : "none";
    });
  });

  // ========== ✏️ AJAX RENAME WITHOUT PAGE RELOAD ==========
  document.querySelectorAll(".rename-form").forEach((form) => {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const videoId = form.dataset.videoId;
      const newTitle = form
        .querySelector("input[name='newTitle']")
        .value.trim();

      if (!newTitle) {
        alert("Title cannot be empty.");
        return;
      }
      const submitBtn = form.querySelector("button");
      submitBtn.disabled = true;
      try {
        const response = await fetch(`/videos/rename/${videoId}`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
          },
          body: JSON.stringify({ newTitle }),
        });

        const res = await response.json();
        if (res.success) {
          showToast("Video renamed successfully!", "success");
          const caption = form
            .closest(".video-container")
            .querySelector(".caption");
          if (caption) caption.textContent = newTitle;
        } else {
          showToast("Rename failed.", "error");
        }
      } catch (err) {
        showToast("An error occurred while renaming.", "error");
      } finally {
        submitBtn.disabled = false;
      }
    });
  });

  setupQueryTabs(); // ← Make sure this is called
});
