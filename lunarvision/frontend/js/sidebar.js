const NAV_ITEMS = [
  { href: "index.html", icon: "🛰️", label: "Dashboard" },
  { href: "analyze.html", icon: "🔬", label: "Analyze" },
  { href: "observations.html", icon: "📁", label: "Saved Observations" },
  { href: "lunar_map.html", icon: "🌗", label: "Lunar Map" },
];

function renderSidebar(activeHref) {
  const links = NAV_ITEMS.map(
    (item) => `
    <a class="nav-link ${item.href === activeHref ? "active" : ""}" href="${item.href}">
      <span>${item.icon}</span><span>${item.label}</span>
    </a>`
  ).join("");

  return `
  <aside class="sidebar">
    <div class="brand">
      <span class="dot">🌘</span>
      <div>
        LunarVision
        <small>SIH26166 · THE UNSOLVED</small>
      </div>
    </div>
    ${links}
    <div class="sidebar-footer">
      Chandrayaan-2 OHRC / TMC-2 / IIRS<br/>
      correspondence &amp; registration prototype
    </div>
  </aside>`;
}

document.addEventListener("DOMContentLoaded", () => {
  const mount = document.getElementById("sidebar-mount");
  if (mount) {
    mount.outerHTML = renderSidebar(document.body.dataset.page);
  }
});
