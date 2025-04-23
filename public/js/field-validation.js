document.addEventListener("DOMContentLoaded", () => {
  const forms = document.querySelectorAll("form");
  const passwordFields = document.querySelectorAll('input[type="password"]');
  const emailInput = document.getElementById("email");
  const emailHint = document.getElementById("email-check");
  const confirmInput = document.getElementById("confirmPassword");
  const passwordInput = document.getElementById("password");
  const passwordWarning = document.getElementById("passwordWarning");
  const mismatchWarning = document.getElementById("passwordMismatch");

  // Show/hide password toggle
  passwordFields.forEach((field) => {
    const wrapper = field.parentElement;
    const toggle = document.createElement("span");
    toggle.classList.add("toggle-password");
    toggle.style.cursor = "pointer";
    toggle.onclick = () => {
      field.type = field.type === "password" ? "text" : "password";
    };
    wrapper.appendChild(toggle);
  });

  // Real-time required field validation
  const requiredInputs = document.querySelectorAll("input[required]");
  requiredInputs.forEach((input) => {
    const alertDiv = input.closest(".form-group")?.querySelector(".field-error");
    input.addEventListener("input", () => {
      const value = input.value.trim();
      if (value === "") {
        input.classList.add("input-error");
        input.classList.remove("input-valid");
        if (alertDiv) alertDiv.style.display = "block";
      } else {
        input.classList.remove("input-error");
        input.classList.add("input-valid");
        if (alertDiv) alertDiv.style.display = "none";
      }
    });
  });

  // Password strength indicator
  const passwordStrength = document.createElement("div");
  passwordStrength.id = "passwordStrength";
  passwordStrength.style.marginTop = "5px";
  passwordStrength.style.fontSize = "13px";
  passwordStrength.style.fontWeight = "bold";
  const currentPath = window.location.pathname;

  if (
    passwordInput &&
    (currentPath.includes("register") ||
      currentPath.includes("profile") ||
      currentPath.includes("reset-password"))
  ) {
    passwordInput.parentElement.appendChild(passwordStrength);
    passwordInput.addEventListener("input", () => {
      const val = passwordInput.value;
      let strength = "";
      let color = "";

      if (
        val.length >= 9 &&
        /[A-Za-z]/.test(val) &&
        /[^A-Za-z0-9]/.test(val) &&
        /[0-9]/.test(val)
      ) {
        strength = "Strong";
        color = "green";
      } else if (
        val.length >= 8 &&
        /[A-Za-z]/.test(val) &&
        /[^A-Za-z0-9]/.test(val)
      ) {
        strength = "Medium";
        color = "orange";
      } else if (val.length === 0) {
        strength = "";
        passwordStrength.textContent = "";
        passwordWarning.style.display = "none";
        return;
      } else {
        strength = "Weak";
        color = "red";
      }

      passwordStrength.textContent = strength
        ? `Password Strength: ${strength}`
        : "";
      passwordStrength.style.color = color;
    });
  }

  // Password match check
  function checkMatch() {
    if (!confirmInput || !passwordInput || !mismatchWarning) return;
    if (confirmInput.value.length === 0) {
      mismatchWarning.style.display = "none";
      confirmInput.classList.remove("input-error");
      return;
    }
    if (confirmInput.value !== passwordInput.value) {
      mismatchWarning.style.display = "block";
      confirmInput.classList.add("input-error");
    } else {
      mismatchWarning.style.display = "none";
      confirmInput.classList.remove("input-error");
    }
  }

  if (confirmInput && passwordInput) {
    confirmInput.addEventListener("input", checkMatch);
    passwordInput.addEventListener("input", checkMatch);
  }

  // ✅ DELETE ACCOUNT FORM: Real-time mismatch check
  const deletePassword = document.getElementById("deletePassword");
  const deleteConfirmPassword = document.getElementById("deleteConfirmPassword");
  const deleteMismatchWarning = document.getElementById("deletePasswordMismatch");

  function checkDeleteMatch() {
    if (!deletePassword || !deleteConfirmPassword || !deleteMismatchWarning) return;

    if (deleteConfirmPassword.value.length === 0) {
      deleteMismatchWarning.style.display = "none";
      deleteConfirmPassword.classList.remove("input-error");
      return;
    }

    if (deletePassword.value !== deleteConfirmPassword.value) {
      deleteMismatchWarning.style.display = "block";
      deleteConfirmPassword.classList.add("input-error");
    } else {
      deleteMismatchWarning.style.display = "none";
      deleteConfirmPassword.classList.remove("input-error");
    }
  }

  if (deletePassword && deleteConfirmPassword) {
    deletePassword.addEventListener("input", checkDeleteMatch);
    deleteConfirmPassword.addEventListener("input", checkDeleteMatch);
  }

  // Email availability check
  if (emailInput && emailInput.dataset.check === "availability") {
    emailInput.addEventListener("input", async () => {
      const email = emailInput.value.trim();
      if (!email.includes("@") || email.length < 5) {
        emailHint.textContent = "";
        emailInput.classList.remove("input-valid", "input-error");
        return;
      }
      const response = await fetch(`/check-email?email=${encodeURIComponent(email)}`);
      const data = await response.json();
      if (data.available) {
        emailHint.textContent = "Email is available";
        emailHint.style.color = "green";
        emailInput.classList.add("input-valid");
        emailInput.classList.remove("input-error");
      } else {
        emailHint.textContent = "Email is already in use";
        emailHint.style.color = "red";
        emailInput.classList.add("input-error");
        emailInput.classList.remove("input-valid");
      }
    });
  }

  // Email match for forgot-password
  const resetEmailInput = document.getElementById("reset-email");
  const resetWarning = document.getElementById("email-match-warning");
  const metaEmail = document.querySelector('meta[name="logged-in-email"]');
  const loggedInEmail = metaEmail ? metaEmail.content : null;
  if (resetEmailInput && loggedInEmail) {
    resetEmailInput.addEventListener("input", () => {
      const inputVal = resetEmailInput.value.trim().toLowerCase();
      const valid = inputVal === loggedInEmail.toLowerCase();
      resetWarning.style.display = valid ? "none" : "block";
      resetEmailInput.classList.toggle("input-error", !valid);
      resetEmailInput.classList.toggle("input-valid", valid);
    });

    const resetForm = resetEmailInput.closest("form");
    if (resetForm) {
      resetForm.addEventListener("submit", function (e) {
        const inputVal = resetEmailInput.value.trim().toLowerCase();
        if (inputVal !== loggedInEmail.toLowerCase()) {
          e.preventDefault();
          resetWarning.style.display = "block";
          resetEmailInput.classList.add("input-error");
        }
      });
    }
  }

  // Final form submission validation
  forms.forEach((form) => {
    form.addEventListener("submit", (e) => {
      const isPasswordForm = form.id === "changePasswordForm";
      let stop = false;

      form.querySelectorAll("input[required]").forEach((input) => {
        const value = input.value.trim();
        const alertDiv = input.closest(".form-group")?.querySelector(".field-error");
        if (value === "") {
          input.classList.add("input-error");
          input.classList.remove("input-valid");
          if (alertDiv) alertDiv.style.display = "block";
          stop = true;
        } else {
          input.classList.remove("input-error");
          input.classList.add("input-valid");
          if (alertDiv) alertDiv.style.display = "none";
        }
      });

      if (isPasswordForm) {
        const passwordRequired = document.getElementById("passwordRequired");
        const currentPasswordInput = document.getElementById("currentPassword");
        const currentPasswordAlert = document.getElementById("currentPasswordRequired");

        if (currentPasswordInput && currentPasswordInput.value.trim() === "") {
          if (currentPasswordAlert) currentPasswordAlert.style.display = "block";
          stop = true;
        }

        if (passwordInput && passwordInput.value.trim() === "") {
          if (passwordRequired) passwordRequired.style.display = "block";
          stop = true;
        } else {
          if (passwordRequired) passwordRequired.style.display = "none";
        }

        if (passwordInput && passwordWarning && passwordInput.value.trim() !== "") {
          const strong =
            passwordInput.value.length >= 9 &&
            /[A-Za-z]/.test(passwordInput.value) &&
            /[^A-Za-z0-9]/.test(passwordInput.value);

          if (!strong) {
            passwordWarning.style.display = "block";
            passwordInput.classList.add("input-error");
            stop = true;
          } else {
            passwordWarning.style.display = "none";
            passwordInput.classList.remove("input-error");
          }
        }

        if (
          passwordInput &&
          confirmInput &&
          mismatchWarning &&
          passwordInput.value !== confirmInput.value
        ) {
          mismatchWarning.style.display = "block";
          confirmInput.classList.add("input-error");
          stop = true;
        }
      }

      if (stop) e.preventDefault();
    });
  });

  // Toggle password form
  const togglePasswordForm = document.getElementById("togglePasswordForm");
  const passwordFormContainer = document.getElementById("passwordFormContainer");
  if (togglePasswordForm && passwordFormContainer) {
    togglePasswordForm.addEventListener("click", () => {
      passwordFormContainer.style.display =
        passwordFormContainer.style.display === "none" ? "block" : "none";
    });
  }
});
