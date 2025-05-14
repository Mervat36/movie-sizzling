document.addEventListener("DOMContentLoaded", () => {
  const containers = document.querySelectorAll(".video-container");

  containers.forEach((container) => {
    const videoTag = container.querySelector("video");
    const source = videoTag.querySelector("source");
    const captionTag = container.querySelector(".caption");
    const downloadBtn = container.querySelector(".btn-download");
    const removeInput = container.querySelector("input[name='videoUrl']");

    const videos = JSON.parse(container.dataset.videos);
    let index = parseInt(container.dataset.index);

    const updateContent = () => {
      const video = videos[index];
      source.src = video.url;
      videoTag.load();
      captionTag.innerHTML = `
    <div>${video.caption}</div>
    ${
      video.startTime && video.endTime
        ? `<div class="time-range">${video.startTime} – ${video.endTime}</div>`
        : ""
    }
  `;
      downloadBtn.href = video.url;
      removeInput.value = video.url;
    };

    container.querySelector(".prev-btn").addEventListener("click", () => {
      index = (index - 1 + videos.length) % videos.length;
      container.dataset.index = index;
      updateContent();
    });

    container.querySelector(".next-btn").addEventListener("click", () => {
      index = (index + 1) % videos.length;
      container.dataset.index = index;
      updateContent();
    });
  });
});
