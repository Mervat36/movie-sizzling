document.addEventListener("DOMContentLoaded", () => {
  // ========== 🔍 FILTER FEATURE ==========
  const input = document.getElementById("historySearchInput");

  input?.addEventListener("input", () => {
    currentSearchTerm = input.value.toLowerCase();
    const wrappers = document.querySelectorAll(".history-table-wrapper");
    let matchCount = 0;

    wrappers.forEach((wrapper) => {
      const videoTitle =
        wrapper.querySelector("input.rename-input")?.value.toLowerCase() || "";
      const queries = [...wrapper.querySelectorAll(".query-tab")].map((tab) =>
        tab.textContent.toLowerCase()
      );
      const matches =
        videoTitle.includes(currentSearchTerm) ||
        queries.some((q) => q.includes(currentSearchTerm));

      wrapper.style.display = matches ? "flex" : "none";
      if (matches) matchCount++;
    });

    // Clear old message
    const oldMessage = document.getElementById("noResultsMessage");
    if (oldMessage) oldMessage.remove();

    // Handle no results
    if (currentSearchTerm && matchCount === 0) {
      const message = document.createElement("div");
      message.id = "noResultsMessage";
      message.className = "empty-state";
      message.innerHTML = `
      <img src="/images/search-empty.png" alt="No results" class="empty-illustration" />
      <h3 class="empty-title">No results found</h3>
      <p class="empty-subtext">Try adjusting your search term above.</p>`;
      document.querySelector(".history-section").appendChild(message);
    }

    currentPage = 1;
    updateVideoPagination(matchCount);
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
  let currentSearchTerm = "";

  function updateVideoPagination() {
    const allWrappers = Array.from(
      document.querySelectorAll(".history-table-wrapper")
    );
    const filteredWrappers = allWrappers.filter((wrapper) => {
      const videoTitle =
        wrapper.querySelector("input.rename-input")?.value.toLowerCase() || "";
      const queries = [...wrapper.querySelectorAll(".query-tab")].map((tab) =>
        tab.textContent.toLowerCase()
      );

      const matches =
        videoTitle.includes(currentSearchTerm) ||
        queries.some((q) => q.includes(currentSearchTerm));
      wrapper.style.display = matches ? "flex" : "none";
      return matches;
    });

    const matchCount = filteredWrappers.length;
    const totalPages = Math.ceil(matchCount / itemsPerPage);

    if (currentPage > totalPages) currentPage = 1;

    // Show only items for the current page
    filteredWrappers.forEach((wrapper, index) => {
      const start = (currentPage - 1) * itemsPerPage;
      const end = currentPage * itemsPerPage;
      wrapper.style.display = index >= start && index < end ? "flex" : "none";
    });

    // Hide others
    allWrappers.forEach((wrapper) => {
      if (!filteredWrappers.includes(wrapper)) {
        wrapper.style.display = "none";
      }
    });

    const container = document.getElementById("video-pagination");
    if (container) {
      container.style.display = matchCount > itemsPerPage ? "flex" : "none";
    }

    renderPaginationControls(totalPages);
  }

  function setupQueryPagination() {
    document.querySelectorAll(".query-card").forEach((queryCard) => {
      const results = Array.from(queryCard.querySelectorAll(".result-card"));
      const itemsPerPage = 1;
      if (results.length <= itemsPerPage) return;

      // Remove old pagination
      const existingPagination = queryCard.querySelector(".pagination");
      if (existingPagination) existingPagination.remove();

      let current = 0;
      const totalPages = Math.ceil(results.length / itemsPerPage);
      const container = document.createElement("div");
      container.className = "pagination";

      const showPage = (page) => {
        results.forEach((res, i) => {
          res.style.display =
            i >= page * itemsPerPage && i < (page + 1) * itemsPerPage
              ? "block"
              : "none";
        });
      };

      const renderButtons = () => {
        container.innerHTML = "";

        const createBtn = (label, page, active = false) => {
          const btn = document.createElement("button");
          btn.innerText = label;
          btn.className = "page-btn";
          if (active) btn.classList.add("active");
          btn.addEventListener("click", () => {
            current = page;
            showPage(current);
            renderButtons();
          });
          return btn;
        };

        const maxButtons = 5;

        if (totalPages <= maxButtons) {
          for (let i = 0; i < totalPages; i++) {
            container.appendChild(createBtn(i + 1, i, i === current));
          }
        } else {
          container.appendChild(createBtn(1, 0, current === 0));

          if (current > 2) {
            const dot = document.createElement("span");
            dot.className = "ellipsis";
            dot.innerText = "• • •";
            container.appendChild(dot);
          }

          const start = Math.max(1, current - 1);
          const end = Math.min(totalPages - 2, current + 1);
          for (let i = start; i <= end; i++) {
            container.appendChild(createBtn(i + 1, i, i === current));
          }

          if (current < totalPages - 3) {
            const dot = document.createElement("span");
            dot.className = "ellipsis";
            dot.innerText = "• • •";
            container.appendChild(dot);
          }

          container.appendChild(
            createBtn(totalPages, totalPages - 1, current === totalPages - 1)
          );
        }
      };

      showPage(current);
      renderButtons();
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
    const container = document.getElementById("video-pagination");
    if (!container) return;

    container.innerHTML = "";

    function createButton(label, page, isActive = false) {
      const btn = document.createElement("button");
      btn.innerText = label;
      btn.className = "page-btn";
      if (isActive) btn.classList.add("active");

      btn.addEventListener("click", () => {
        currentPage = page;
        updateVideoPagination();
        input.value = currentSearchTerm;
      });

      return btn;
    }

    const maxButtons = 5;
    const pageButtons = [];

    if (totalPages <= maxButtons) {
      for (let i = 1; i <= totalPages; i++) {
        pageButtons.push(createButton(i, i, i === currentPage));
      }
    } else {
      // Always show first
      pageButtons.push(createButton(1, 1, currentPage === 1));

      if (currentPage > 3) {
        const ellipsis = document.createElement("span");
        ellipsis.innerText = "• • •";
        ellipsis.className = "ellipsis";
        pageButtons.push(ellipsis);
      }

      // Middle numbers
      const start = Math.max(2, currentPage - 1);
      const end = Math.min(totalPages - 1, currentPage + 1);
      for (let i = start; i <= end; i++) {
        if (i !== 1 && i !== totalPages) {
          pageButtons.push(createButton(i, i, i === currentPage));
        }
      }

      if (currentPage < totalPages - 2) {
        const ellipsis = document.createElement("span");
        ellipsis.innerText = "• • •";
        ellipsis.className = "ellipsis";
        pageButtons.push(ellipsis);
      }

      // Always show last
      pageButtons.push(
        createButton(totalPages, totalPages, currentPage === totalPages)
      );
    }

    pageButtons.forEach((btn) => container.appendChild(btn));
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
          modal.classList.add("hidden");
          const spinner = document.getElementById("loading-spinner");
          if (spinner) spinner.style.display = "flex";
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
              if (spinner) spinner.style.display = "none";
              return;
            } finally {
              if (spinner) spinner.style.display = "none";
            }
            if (!res.ok || !json.success) {
              showToast(`Failed to delete ${deleteType}.`, "error");
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
                    queryList.innerHTML = `
                    <div class="empty-query-message">
                      <img src="/images/search-empty.png" alt="No queries" />
                      <h4>No Queries Found</h4>
                      <p>You haven't searched any queries for this video yet. Try exploring a scene!</p>
                    </div>
                    `;
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
                queryList.innerHTML = queryList.innerHTML = `
                  <div class="empty-query-message">
                    <img src="/images/search-empty.png" alt="No queries" />
                    <h4>No Queries Found</h4>
                    <p>You haven't searched any queries for this video yet. Try exploring a scene!</p>
                  </div>  
                  `;
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
                `${deleteType.charAt(0).toUpperCase() + deleteType.slice(1)
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
  // Scroll-to-top button logic
  const scrollBtn = document.getElementById("scrollToTop");

  window.addEventListener("scroll", () => {
    if (window.scrollY > 400) {
      scrollBtn.style.display = "block";
    } else {
      scrollBtn.style.display = "none";
    }
  });

  scrollBtn.addEventListener("click", () => {
    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  });

  setupQueryTabs(); // ← Make sure this is called
});

function openRenameModal(videoId, formElement) {
  const newTitle = formElement.querySelector('input[name="newTitle"]').value;
  const isHidden = formElement.querySelector('input[name="isHidden"]').checked;

  // Set values in modal hidden fields
  document.getElementById('confirmNewTitle').value = newTitle;
  document.getElementById('confirmIsHidden').value = isHidden ? "true" : "";

  const renameForm = document.getElementById('confirmRenameForm');
  renameForm.action = `/videos/rename/${videoId}`;

  // Show modal
  document.getElementById("renameModal").classList.remove("hidden");
}

function closeRenameModal() {
  document.getElementById("renameModal").classList.add("hidden");
}

let tempCheckbox = null;

function onHideClick(event, videoId, checkbox) {
  event.preventDefault(); // stop the checkbox from toggling immediately

  tempCheckbox = checkbox;

  const form = checkbox.closest("form");
  const newTitle = form.querySelector('input[name="newTitle"]').value;
  const intendedState = !checkbox.checked;

  document.getElementById("hiddenTitleValue").value = newTitle;
  document.getElementById("hiddenIsHiddenValue").value = intendedState ? "true" : "";
  document.getElementById("confirmHideForm").action = `/videos/rename/${videoId}`;
  document.getElementById("confirmRenameForm").action = `/videos/rename/${videoId}`;
  document.getElementById("hideModal").classList.remove("hidden");
}
// ========== ✅ CONFIRM RENAME SUBMIT (MODAL) ==========
document.getElementById("confirmRenameForm").addEventListener("submit", async function (e) {
  e.preventDefault();

  const form = e.target;
  const videoId = form.action.split("/").pop(); // Extract ID
  const newTitle = document.getElementById("confirmNewTitle").value.trim();
  const isHidden = document.getElementById("confirmIsHidden").value === "true";

  if (!newTitle) {
    alert("Title cannot be empty.");
    return;
  }

  const submitBtn = form.querySelector("button[type='submit']");
  submitBtn.disabled = true;

  try {
    const response = await fetch(`/videos/rename/${videoId}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({ newTitle, isHidden }),
    });

    const res = await response.json();
    if (res.success) {
      showToast("Video renamed successfully!", "success");

      // Update the input title value on the page immediately
      const renameInput = document.querySelector(`form[data-video-id="${videoId}"] input[name="newTitle"]`);
      if (renameInput) renameInput.value = newTitle;

      // Update checkbox state
      const checkbox = document.querySelector(`form[data-video-id="${videoId}"] input[name="isHidden"]`);
      if (checkbox) checkbox.checked = isHidden;
    } else {
      showToast("Rename failed.", "error");
    }
  } catch (err) {
    showToast("An error occurred while renaming.", "error");
  } finally {
    submitBtn.disabled = false;
    closeRenameModal();
  }
});

document.getElementById("cancelHideToggle").onclick = () => {
  document.getElementById("hideModal").classList.add("hidden");

  // Revert checkbox to original state
  if (tempCheckbox) {
    tempCheckbox.checked = !tempCheckbox.checked;
    tempCheckbox = null;
  }
};

window.onHideClick = onHideClick;
