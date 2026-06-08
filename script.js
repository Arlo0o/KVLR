const header = document.querySelector("[data-header]");
const navToggle = document.querySelector(".nav-toggle");
const navLinks = document.querySelectorAll(".nav-links a");

navToggle?.addEventListener("click", () => {
  const isOpen = header.classList.toggle("open");
  navToggle.setAttribute("aria-expanded", String(isOpen));
});

navLinks.forEach((link) => {
  link.addEventListener("click", () => {
    header.classList.remove("open");
    navToggle?.setAttribute("aria-expanded", "false");
  });
});

const demoVideo = document.querySelector(".video-frame video");

if (demoVideo) {
  demoVideo.muted = true;
  demoVideo.play().catch(() => {
    // Autoplay can still be delayed by browser policy until the media is near the viewport.
  });
}

const figures = {
  compare: {
    src: "public/assets/vis_compare.webp",
    alt: "Qualitative comparison on KASA.",
    width: 3271,
    height: 1343,
    caption:
      "KVLR better preserves tool states and action-consistent evolution under clutter and poor illumination.",
  },
  generalize: {
    src: "public/assets/vis_generalize.webp",
    alt: "Generalization results with what-if action control and transfer to unseen scenes.",
    width: 3216,
    height: 914,
    caption:
      "Generalization results show controllable synthesis under user-specified actions and transfer to unseen scenes.",
  },
  action: {
    src: "public/assets/vis_action.webp",
    alt: "Visualization of action-conditioned control signals.",
    width: 3814,
    height: 1578,
    caption:
      "Action visualization reveals how articulated controls map into image-aligned control channels.",
  },
  motion: {
    src: "public/assets/vis_motion_plot.webp",
    alt: "Motion plot for surgical action trajectories.",
    width: 4454,
    height: 1546,
    caption:
      "Motion plots highlight heterogeneous dynamics across transport motion, local manipulation, and stable regions.",
  },
  diverse: {
    src: "public/assets/fig_more_diverse1.webp",
    alt: "Additional diverse generation results from KVLR.",
    width: 3169,
    height: 4556,
    caption:
      "Additional diverse generation examples cover different surgical actions and visual conditions.",
  },
  ablation: {
    src: "public/assets/supple_abl_arc.webp",
    alt: "Architecture ablation qualitative comparison.",
    width: 3550,
    height: 2344,
    caption:
      "Architecture ablations show the contribution of KVA-Field lifting and hierarchical routing.",
  },
};

const galleryPanel = document.querySelector("[data-gallery-panel]");
const galleryImage = galleryPanel?.querySelector("img");
const galleryCaption = galleryPanel?.querySelector("figcaption");
const galleryTabs = document.querySelectorAll(".gallery-tabs button");

galleryTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const key = tab.dataset.figure;
    const figure = figures[key];
    if (!figure || !galleryImage || !galleryCaption) return;

    galleryTabs.forEach((item) => item.setAttribute("aria-selected", "false"));
    tab.setAttribute("aria-selected", "true");

    galleryImage.src = figure.src;
    galleryImage.alt = figure.alt;
    galleryImage.width = figure.width;
    galleryImage.height = figure.height;
    galleryCaption.textContent = figure.caption;
  });
});

const lightbox = document.querySelector("[data-lightbox]");
const lightboxImage = lightbox?.querySelector("img");
const closeLightbox = lightbox?.querySelector(".lightbox-close");

document.querySelectorAll(".wide-figure img").forEach((image) => {
  image.addEventListener("click", () => {
    if (!lightbox || !lightboxImage) return;
    lightboxImage.src = image.currentSrc || image.src;
    lightboxImage.alt = image.alt;
    lightbox.hidden = false;
    closeLightbox?.focus();
  });
});

function hideLightbox() {
  if (!lightbox || !lightboxImage) return;
  lightbox.hidden = true;
  lightboxImage.removeAttribute("src");
}

closeLightbox?.addEventListener("click", hideLightbox);
lightbox?.addEventListener("click", (event) => {
  if (event.target === lightbox) hideLightbox();
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && lightbox && !lightbox.hidden) {
    hideLightbox();
  }
});
