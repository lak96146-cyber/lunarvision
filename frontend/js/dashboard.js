async function loadDashboard() {
  let observations = [];
  try {
    observations = await Api.listObservations();
  } catch (e) {
    console.error(e);
  }

  document.getElementById("stat-total").textContent = observations.length;

  if (observations.length > 0) {
    const avgRatio = observations.reduce((s, o) => s + (o.inlier_ratio || 0), 0) / observations.length;
    const rmseVals = observations.filter((o) => o.rmse_px != null).map((o) => o.rmse_px);
    const avgRmse = rmseVals.length ? rmseVals.reduce((s, v) => s + v, 0) / rmseVals.length : null;
    const pairs = new Set(observations.map((o) => `${o.source_sensor}↔${o.reference_sensor}`));

    document.getElementById("stat-ratio").textContent = (avgRatio * 100).toFixed(1) + "%";
    document.getElementById("stat-rmse").textContent = avgRmse != null ? avgRmse.toFixed(2) : "—";
    document.getElementById("stat-pairs").textContent = pairs.size;
  } else {
    document.getElementById("stat-ratio").textContent = "—";
    document.getElementById("stat-rmse").textContent = "—";
    document.getElementById("stat-pairs").textContent = "0";
  }

  const tbody = document.querySelector("#recent-table tbody");
  if (observations.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" style="color:var(--text-dim)">No observations saved yet. Run your first analysis →</td></tr>`;
    return;
  }

  tbody.innerHTML = observations
    .slice(0, 8)
    .map(
      (o) => `
    <tr>
      <td>#${o.id}</td>
      <td>${o.source_sensor} → ${o.reference_sensor}</td>
      <td>${o.detector.toUpperCase()}</td>
      <td>${(o.inlier_ratio * 100).toFixed(1)}%</td>
      <td>${o.rmse_px != null ? o.rmse_px.toFixed(2) : "—"}</td>
      <td><span class="badge ${o.is_synthetic ? "synthetic" : "real"}">${o.is_synthetic ? "SYNTHETIC" : "REAL"}</span></td>
      <td><a href="observations.html?id=${o.id}">View →</a></td>
    </tr>`
    )
    .join("");
}

loadDashboard();
