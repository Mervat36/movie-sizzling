// ========== Sidebar Navigation and Tab Switching ==========
const savedTab = localStorage.getItem("admin-active-tab");
if (savedTab) {
    document.querySelector(`.nav-link[href="#${savedTab}"]`)?.click();
}
const navLinks = document.querySelectorAll(".sidebar-nav .nav-link");
const sections = document.querySelectorAll(".admin-section");
navLinks.forEach(link => {
    link.addEventListener("click", e => {
        const href = e.currentTarget.getAttribute("href");
        if (!href.startsWith("#")) {
            window.location.href = href;
            return;
        }
        e.preventDefault();
        localStorage.setItem("admin-active-tab", href.substring(1));
        navLinks.forEach(l => l.classList.remove("active"));
        e.currentTarget.classList.add("active");
        sections.forEach(sec => sec.classList.add("hidden"));
        document.getElementById(href.substring(1))?.classList.remove("hidden");
    });
});
document.getElementById("dashboard").classList.remove("hidden");

// ========== Theme Switching ==========
const adminThemeToggle = document.getElementById("adminThemeToggle");
const body = document.body;
const savedTheme = localStorage.getItem("admin-theme");
if (savedTheme === "dark") {
    body.classList.add("dark-mode");
    adminThemeToggle.checked = true;
}
adminThemeToggle.addEventListener("change", () => {
    if (adminThemeToggle.checked) {
        body.classList.add("dark-mode");
        localStorage.setItem("admin-theme", "dark");
    } else {
        body.classList.remove("dark-mode");
        localStorage.setItem("admin-theme", "light");
    }
});

// ========== Ban User Modal ==========
function openBanModal(userId, userName) {
    const modal = document.getElementById('banModal');
    const form = document.getElementById('banForm');
    form.dataset.userId = userId;
    document.getElementById('banModalTitle').innerText = `Ban ${userName}`;
    modal.classList.remove('hidden');
}
function closeBanModal() {
    document.getElementById('banModal').classList.add('hidden');
}
async function submitBanForm(event) {
    event.preventDefault();
    const form = event.target;
    const userId = form.dataset.userId;
    const banDate = form.querySelector("#banDatePicker").value;
    if (!banDate) return showToast("Please select a date.", "error");
    showLoader();
    try {
        const res = await fetch(`/admin/ban-user/${userId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ banUntil: banDate })
        });
        if (res.ok) {
            const card = document.querySelector(`.user-card button[onclick*="${userId}"]`)?.closest(".user-card");
            if (card) {
                const actions = card.querySelector(".card-actions");
                actions.querySelector(".btn-ban")?.remove();
                const unbanBtn = document.createElement("button");
                unbanBtn.className = "btn-unban";
                unbanBtn.textContent = "Unban";
                unbanBtn.onclick = () => unbanUser(userId, unbanBtn);
                actions.insertBefore(unbanBtn, actions.querySelector("form"));
                const info = card.querySelector(".user-info");
                const banText = document.createElement("p");
                banText.className = "user-ban";
                banText.textContent = `Banned until ${new Date(banDate).toLocaleDateString()}`;
                info.appendChild(banText);
            }
            closeBanModal();
            showToast("User banned until " + banDate, "success");
        } else showToast("Failed to ban user.", "error");
    } catch (err) {
        console.error("Ban error:", err);
        showToast("Server error while banning user.", "error");
    } finally {
        hideLoader();
    }
}

// ========== Unban User ==========
async function unbanUser(userId, button) {
    showLoader();
    try {
        const response = await fetch(`/admin/unban/${userId}`, { method: "POST" });
        if (response.ok) {
            const card = button.closest(".user-card");
            card.querySelector(".user-ban")?.remove();
            button.remove();
            const newBanBtn = document.createElement("button");
            newBanBtn.className = "btn-ban";
            newBanBtn.textContent = "Ban";
            newBanBtn.onclick = () => openBanModal(userId, card.querySelector(".user-name").textContent);
            card.querySelector(".card-actions").prepend(newBanBtn);
            showToast("User unbanned successfully.");
        } else showToast("Failed to unban user.", "error");
    } catch (error) {
        console.error("Unban error:", error);
        showToast("An error occurred while unbanning.", "error");
    } finally {
        hideLoader();
    }
}

// ========== Role Update ==========
function openRoleModal(userId, userName, isAdmin) {
    const modal = document.getElementById("roleModal");
    const form = document.getElementById("roleForm");
    document.getElementById("roleModalTitle").innerText = isAdmin ? `Remove admin role from ${userName}?` : `Make ${userName} an admin?`;
    form.action = isAdmin ? `/admin/remove-admin/${userId}` : `/admin/make-admin/${userId}`;
    modal.classList.remove("hidden");
}
function closeRoleModal() {
    document.getElementById("roleModal").classList.add("hidden");
}
async function submitRoleForm(event) {
    event.preventDefault();
    const form = event.target;
    const actionUrl = form.action;
    const isRemoving = actionUrl.includes("/remove-admin/");
    const userId = actionUrl.split("/").pop();
    showLoader();
    try {
        const res = await fetch(actionUrl, { method: "POST" });
        if (res.ok) {
            const card = [...document.querySelectorAll(".user-card")].find(c =>
                c.querySelector(".btn-role")?.getAttribute("onclick")?.includes(userId)
            );
            if (card) {
                const roleText = card.querySelector(".user-role");
                const btn = card.querySelector(".btn-role");
                const newStatus = isRemoving ? "User" : "Admin";
                roleText.textContent = newStatus;
                roleText.className = `user-role ${isRemoving ? "user" : "admin"}`;
                btn.textContent = isRemoving ? "Make Admin" : "Remove Admin";
                btn.setAttribute("onclick", `openRoleModal('${userId}', '${card.querySelector(".user-name").textContent}', ${!isRemoving})`);
            }
            closeRoleModal();
            showToast(isRemoving ? "Admin role removed." : "User promoted to admin.", "success");
        } else showToast("Failed to update role.", "error");
    } catch (err) {
        console.error("Role update error:", err);
        showToast("Server error while updating role.", "error");
    } finally {
        hideLoader();
    }
}

// ========== Delete User ==========
function openDeleteModal(userId, userName) {
    const modal = document.getElementById("deleteUserModal");
    const form = document.getElementById("deleteUserForm");
    document.getElementById("deleteUserName").textContent = userName;
    form.action = `/admin/delete-user/${userId}`;
    modal.classList.remove("hidden");
}
function closeDeleteModal() {
    document.getElementById("deleteUserModal").classList.add("hidden");
}
document.getElementById("deleteUserForm")?.addEventListener("submit", async function (e) {
    e.preventDefault();
    const form = e.target;
    showLoader();
    try {
        const res = await fetch(form.action, { method: "POST" });
        if (res.ok) {
            const userId = form.action.split("/").pop();
            document.querySelector(`.user-card button[onclick*="${userId}"]`)?.closest(".user-card")?.remove();
            closeDeleteModal();
            showToast("User deleted successfully.", "success");
        } else showToast("Failed to delete user.", "error");
    } catch (err) {
        console.error("Delete error:", err);
        showToast("Server error while deleting user.", "error");
    } finally {
        hideLoader();
    }
});

// ========== Delete Video ==========
function submitDeleteVideoForm(event, videoId) {
    event.preventDefault();
    showLoader();
    fetch(`/admin/delete-video/${videoId}`, { method: "POST" })
        .then(res => {
            if (res.ok) {
                document.querySelector(`.video-card form[onsubmit*="${videoId}"]`)?.closest(".video-card")?.remove();
                showToast("Video deleted successfully.", "success");
            } else {
                showToast("Failed to delete video.", "error");
            }
        })
        .catch(err => {
            console.error("Delete video error:", err);
            showToast("Server error while deleting video.", "error");
        })
        .finally(() => {
            hideLoader();
        });
}
let videoToDeleteId = null;

function openDeleteVideoModal(videoId) {
    videoToDeleteId = videoId;
    document.getElementById("deleteVideoModal").classList.remove("hidden");
}

function closeDeleteVideoModal() {
    document.getElementById("deleteVideoModal").classList.add("hidden");
    videoToDeleteId = null;
}

document.addEventListener("DOMContentLoaded", () => {
    const videoForm = document.getElementById("deleteVideoForm");
    if (!videoForm) return;

    videoForm.addEventListener("submit", async function (e) {
        e.preventDefault();
        if (!videoToDeleteId) return;

        showLoader();
        try {
            const res = await fetch(`/admin/delete-video/${videoToDeleteId}`, { method: "POST" });
            if (res.ok) {
                const videoCard = document.querySelector(`.video-card button[onclick*="${videoToDeleteId}"]`)?.closest(".video-card");
                if (videoCard) videoCard.remove();
                showToast("Video deleted successfully.", "success");
            } else {
                showToast("Failed to delete video.", "error");
            }
        } catch (err) {
            console.error("Delete video error:", err);
            showToast("Server error while deleting video.", "error");
        } finally {
            hideLoader();
            closeDeleteVideoModal();
        }
    });
});


// ========== Search Inputs ==========
document.getElementById("userSearchInput")?.addEventListener("input", () => {
    const searchTerm = document.getElementById("userSearchInput").value.toLowerCase();
    document.querySelectorAll(".user-card").forEach(card => {
        const name = card.querySelector(".user-name")?.textContent.toLowerCase() || "";
        const email = card.querySelector(".user-email")?.textContent.toLowerCase() || "";
        card.style.display = name.includes(searchTerm) || email.includes(searchTerm) ? "block" : "none";
    });
});
document.getElementById("videoSearchInput")?.addEventListener("input", () => {
    const searchTerm = document.getElementById("videoSearchInput").value.toLowerCase();
    document.querySelectorAll(".video-card").forEach(card => {
        const title = card.querySelector(".video-title")?.textContent.toLowerCase() || "";
        const uploader = card.querySelector(".video-uploader")?.textContent.toLowerCase() || "";
        card.style.display = title.includes(searchTerm) || uploader.includes(searchTerm) ? "block" : "none";
    });
});

// ========== Toast & Loader ==========
function showLoader() {
    document.getElementById("globalLoader")?.classList.remove("hidden");
}
function hideLoader() {
    document.getElementById("globalLoader")?.classList.add("hidden");
}
function showToast(message, type = "success") {
    if (!message || typeof message !== "string" || message.trim() === "") return; // Ensure no empty toast

    const toast = document.getElementById("toast");
    if (!toast) return;

    toast.textContent = message;
    toast.className = `toast show ${type}`;

    setTimeout(() => {
        toast.className = "toast";
        toast.textContent = "";
    }, 4000);
}




// ========== Flatpickr Setup ==========
flatpickr("#banDatePicker", {
    dateFormat: "Y-m-d",
    minDate: "today",
});
