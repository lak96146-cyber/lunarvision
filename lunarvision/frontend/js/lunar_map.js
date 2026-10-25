async function loadChecklist() {
  const observations = await Api.listObservations();
  const container = document.getElementById("obs-checklist");

  if (observations.length === 0) {
    container.innerHTML = `<div class="empty-state" style="padding:10px 0;">No saved observations yet — analyze and save at least two before building a mosaic.</div>`;
    return;
  }

  container.innerHTML = observations
    .map(
      (o) => `
    <label class="checkbox-row" style="margin-bottom:8px;">
      <input type="checkbox" class="obs-check" value="${o.id}" />
      #${o.id} — ${o.source_sensor}→${o.reference_sensor} (${o.detector.toUpperCase()}, ${(o.inlier_ratio * 100).toFixed(0)}% inliers)
      <span class="badge ${o.is_synthetic ? "synthetic" : "real"}">${o.is_synthetic ? "SYN" : "REAL"}</span>
    </label>`
    )
    .join("");
}

document.getElementById("btn-build-mosaic").addEventListener("click", async () => {
  const ids = Array.from(document.querySelectorAll(".obs-check:checked")).map((el) => parseInt(el.value, 10));
  const status = document.getElementById("mosaic-status");
  if (ids.length < 1) {
    alert("Select at least one observation (two or more recommended, sharing the same reference image).");
    return;
  }
  status.textContent = "Building mosaic…";
  try {
    const res = await Api.buildMosaic(ids);
    document.getElementById("mosaic-result").style.display = "block";
    document.getElementById("m-pieces").textContent = res.stats.num_pieces;
    document.getElementById("m-size").textContent = `${res.stats.canvas_width}×${res.stats.canvas_height}`;
    const overlapPct = res.stats.coverage_pixels
      ? ((res.stats.overlap_pixels / res.stats.coverage_pixels) * 100).toFixed(1) + "%"
      : "0%";
    document.getElementById("m-overlap").textContent = overlapPct;
    document.getElementById("m-skipped").textContent = res.skipped.length;
    document.getElementById("mosaic-image").src = staticUrl(res.mosaic_image) + `?t=${Date.now()}`;
    document.getElementById("m-contrib").textContent = res.contributing_observations.map((id) => "#" + id).join(", ");
    status.textContent = "✅ Done";
  } catch (e) {
    status.textContent = "❌ Failed: " + e.message;
  }
});

loadChecklist();
