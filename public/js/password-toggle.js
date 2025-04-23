
document.addEventListener("DOMContentLoaded", () => {
  const passwordFields = document.querySelectorAll('input[type="password"]');

  passwordFields.forEach(field => {
    // 1. Wrap the input inside a div
    const wrapper = document.createElement("div");
    wrapper.classList.add("password-wrapper");
    field.parentElement.insertBefore(wrapper, field);
    wrapper.appendChild(field);
    // 2. Add toggle icon after the input
    const toggle = document.createElement("span");
    toggle.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" height="20" width="20" viewBox="0 0 24 24" fill="none" stroke="#999" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-eye-off"><path d="M17.94 17.94A10.94 10.94 0 0 1 12 20C7.58 20 3.73 17.11 1.82 12.7a.998.998 0 0 1 0-.81A10.99 10.99 0 0 1 6.1 6.1M9.9 9.9A3 3 0 0 1 14.1 14.1M12 12m-9 0c2-4 6-7 9-7s7 3 9 7c-2 4-6 7-9 7s-7-3-9-7z"></path></svg>';
    toggle.classList.add("password-toggle-icon");
    toggle.style.cursor = "pointer";
    wrapper.appendChild(toggle);
    // 3. Toggle password visibility and change icon on click
    toggle.addEventListener("click", () => {
      if (field.type === "password") {
        field.type = "text";
        toggle.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" height="20" width="20" viewBox="0 0 24 24" fill="none" stroke="#999" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-eye"><circle cx="12" cy="12" r="3"/><path d="M21 12c-1.73-4.09-5.64-7-9-7s-7.27 2.91-9 7c1.73 4.09 5.64 7 9 7s7.27-2.91 9-7z"/></svg>';
      } else {
        field.type = "password";
        toggle.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" height="20" width="20" viewBox="0 0 24 24" fill="none" stroke="#999" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-eye-off"><path d="M17.94 17.94A10.94 10.94 0 0 1 12 20C7.58 20 3.73 17.11 1.82 12.7a.998.998 0 0 1 0-.81A10.99 10.99 0 0 1 6.1 6.1M9.9 9.9A3 3 0 0 1 14.1 14.1M12 12m-9 0c2-4 6-7 9-7s7 3 9 7c-2 4-6 7-9 7s-7-3-9-7z"></path></svg>';
      }
    });
    
  });
});
