let allObservations = [];

async function loadObservations() {
  allObservations = await Api.listObservations();
  const tbody = document.querySelector("#obs-table tbody");

  if (allObservations.length === 0) {
    tbody.innerHTML = `<tr><td colspan="5" style="color:var(--text-dim)">No saved observations yet.</td></tr>`;
    return;
  }

  tbody.innerHTML = allObservations
    .map(
      (o) => `
    <tr style="cursor:pointer" onclick="showDetail(${o.id})">
      <td><span class="badge ${o.is_synthetic ? "synthetic" : "real"}">${o.is_synthetic ? "SYN" : "REAL"}</span></td>
      <td>#${o.id}</td>
      <td>${o.source_sensor}→${o.reference_sensor}</td>
      <td>${(o.inlier_ratio * 100).toFixed(1)}%</td>
      <td>${o.detector.toUpperCase()}</td>
    </tr>`
    )
    .join("");

  const params = new URLSearchParams(window.location.search);
  const idParam = params.get("id");
  if (idParam) {
    showDetail(parseInt(idParam, 10));
  } else if (allObservations.length > 0) {
    showDetail(allObservations[0].id);
  }
}

function showDetail(id) {
  const o = allObservations.find((x) => x.id === id);
  if (!o) return;
  const panel = document.getElementById("detail-panel");
  const t = Date.now();
  panel.innerHTML = `
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
      <h3 style="margin:0;">Observation #${o.id}</h3>
      <span class="badge ${o.is_synthetic ? "synthetic" : "real"}">${o.is_synthetic ? "SYNTHETIC" : "REAL DATA"}</span>
    </div>
    <p style="color:var(--text-dim); font-size:13px;">${new Date(o.created_at).toLocaleString()}</p>
    <div class="grid grid-4" style="margin-bottom:16px;">
      <div class="card stat-card"><div class="label">Matches</div><div class="value accent">${o.total_matches}</div></div>
      <div class="card stat-card"><div class="label">Inliers</div><div class="value green">${o.inliers}</div></div>
      <div class="card stat-card"><div class="label">Ratio</div><div class="value">${(o.inlier_ratio * 100).toFixed(1)}%</div></div>
      <div class="card stat-card"><div class="label">RMSE (px)</div><div class="value">${o.rmse_px != null ? o.rmse_px.toFixed(2) : "—"}</div></div>
    </div>
    <p style="font-size:13px;"><b>Sensors:</b> <span class="badge sensor">${o.source_sensor}</span> → <span class="badge sensor">${o.reference_sensor}</span> &nbsp; <b>Detector:</b> ${o.detector.toUpperCase()}</p>
    <div class="grid grid-2">
      <div>
        <p style="font-size:12.5px; color:var(--text-dim); margin-bottom:6px;">Correspondence</p>
        <div class="result-img-wrap"><img src="${staticUrl("processed/" + o.correspondence_image)}?t=${t}" /></div>
      </div>
      <div>
        <p style="font-size:12.5px; color:var(--text-dim); margin-bottom:6px;">Alignment Overlay</p>
        <div class="result-img-wrap"><img src="${staticUrl("processed/" + o.overlay_image)}?t=${t}" /></div>
      </div>
    </div>
  `;
}

loadObservations();
