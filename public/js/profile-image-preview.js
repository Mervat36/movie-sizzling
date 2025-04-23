document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("imageUpload");
    const previewImage = document.getElementById("profilePicPreview");
    // 1. When file input changes, show preview
    if (fileInput && previewImage) {
      fileInput.addEventListener("change", function () {
        const file = fileInput.files[0];
        if (file) {
          const imageUrl = URL.createObjectURL(file);
          previewImage.src = imageUrl;
        }
      });
    }
  });
  