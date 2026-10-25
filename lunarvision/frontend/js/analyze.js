let currentSyntheticPairId = null;
let currentResult = null;
let sourceFile = null;
let referenceFile = null;

const useSyntheticCheckbox = document.getElementById("use-synthetic");
const syntheticControls = document.getElementById("synthetic-controls");
const uploadControls = document.getElementById("upload-controls");
const btnGenerateSynthetic = document.getElementById("btn-generate-synthetic");
const previewRow = document.getElementById("preview-row");

function toggleMode() {
  const synthetic = useSyntheticCheckbox.checked;
  syntheticControls.style.display = synthetic ? "grid" : "none";
  btnGenerateSynthetic.style.display = synthetic ? "inline-flex" : "none";
  uploadControls.style.display = synthetic ? "none" : "grid";
}
useSyntheticCheckbox.addEventListener("change", toggleMode);
toggleMode();

btnGenerateSynthetic.addEventListener("click", async () => {
  btnGenerateSynthetic.disabled = true;
  btnGenerateSynthetic.textContent = "Generating…";
  try {
    const params = {
      rotation_deg: document.getElementById("syn-rotation").value,
      scale: document.getElementById("syn-scale").value,
      tx: document.getElementById("syn-tx").value,
      ty: document.getElementById("syn-ty").value,
    };
    const res = await Api.generateSyntheticPair(params);
    currentSyntheticPairId = res.pair_id;
    document.getElementById("preview-source").src = staticUrl(res.source_image) + `?t=${Date.now()}`;
    document.getElementById("preview-reference").src = staticUrl(res.reference_image) + `?t=${Date.now()}`;
    previewRow.style.display = "grid";
  } catch (e) {
    alert("Failed to generate synthetic pair: " + e.message);
  } finally {
    btnGenerateSynthetic.disabled = false;
    btnGenerateSynthetic.textContent = "🎲 Generate Synthetic Pair";
  }
});

function setupUploadBox(boxId, inputId, isSource) {
  const box = document.getElementById(boxId);
  const input = document.getElementById(inputId);
  box.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    const file = input.files[0];
    if (!file) return;
    if (isSource) sourceFile = file; else referenceFile = file;
    const url = URL.createObjectURL(file);
    box.innerHTML = `<img src="${url}" />`;
  });
}
setupUploadBox("source-upload-box", "source-file", true);
setupUploadBox("reference-upload-box", "reference-file", false);

const pipelineStatus = document.getElementById("pipeline-status");
const analyzeStatus = document.getElementById("analyze-status");
const stepOrder = ["preprocess", "detect", "match", "ransac", "register", "metrics"];

async function animatePipeline() {
  pipelineStatus.style.display = "block";
  const spans = pipelineStatus.querySelectorAll("span");
  spans.forEach((s) => s.classList.remove("done"));
  for (let i = 0; i < stepOrder.length; i++) {
    await new Promise((res) => setTimeout(res, 220));
    spans[i].classList.add("done");
  }
}

document.getElementById("btn-analyze").addEventListener("click", async () => {
  const isSynthetic = useSyntheticCheckbox.checked;
  if (isSynthetic && !currentSyntheticPairId) {
    alert("Generate a synthetic pair first.");
    return;
  }
  if (!isSynthetic && (!sourceFile || !referenceFile)) {
    alert("Upload both a source and reference image.");
    return;
  }

  const btn = document.getElementById("btn-analyze");
  btn.disabled = true;
  analyzeStatus.textContent = "Running pipeline…";
  animatePipeline();

  try {
    const result = await Api.runAnalysis({
      sourceSensor: document.getElementById("source-sensor").value,
      referenceSensor: document.getElementById("reference-sensor").value,
      detector: document.getElementById("detector").value,
      isSynthetic,
      syntheticPairId: currentSyntheticPairId,
      sourceFile,
      referenceFile,
    });
    currentResult = result;
    renderResults(result);
    analyzeStatus.textContent = "✅ Done";
  } catch (e) {
    analyzeStatus.textContent = "❌ Failed";
    alert("Analysis failed: " + e.message);
  } finally {
    btn.disabled = false;
  }
});

function renderResults(result) {
  document.getElementById("results-section").style.display = "block";
  document.getElementById("r-total").textContent = result.total_matches;
  document.getElementById("r-inliers").textContent = result.inliers;
  document.getElementById("r-ratio").textContent = (result.inlier_ratio * 100).toFixed(1) + "%";
  document.getElementById("r-rmse").textContent = result.rmse_px != null ? result.rmse_px.toFixed(3) : "—";
  document.getElementById("r-kp-src").textContent = result.keypoints_source;
  document.getElementById("r-kp-ref").textContent = result.keypoints_reference;
  document.getElementById("r-gt").textContent = result.ground_truth_corner_error_px != null
    ? result.ground_truth_corner_error_px.toFixed(3)
    : "N/A (real data — no ground truth)";
  document.getElementById("r-time").textContent = result.processing_time_sec;

  const t = Date.now();
  document.getElementById("r-correspondence").src = staticUrl(result.image_urls.correspondence_image) + `?t=${t}`;
  document.getElementById("r-overlay").src = staticUrl(result.image_urls.overlay_image) + `?t=${t}`;
  document.getElementById("cmp-bottom").src = staticUrl(result.image_urls.reference_preprocessed) + `?t=${t}`;
  document.getElementById("cmp-top").src = staticUrl(result.image_urls.warped_image) + `?t=${t}`;

  document.getElementById("results-section").scrollIntoView({ behavior: "smooth" });
}

document.getElementById("slider-range").addEventListener("input", (e) => {
  const val = e.target.value;
  document.getElementById("cmp-top").style.clipPath = `inset(0 ${100 - val}% 0 0)`;
});

document.getElementById("btn-save").addEventListener("click", async () => {
  if (!currentResult) return;
  const saveStatus = document.getElementById("save-status");
  saveStatus.textContent = "Saving…";
  try {
    const payload = {
      source_sensor: currentResult.source_sensor,
      reference_sensor: currentResult.reference_sensor,
      detector: currentResult.detector,
      is_synthetic: currentResult.is_synthetic,
      total_matches: currentResult.total_matches,
      inliers: currentResult.inliers,
      inlier_ratio: currentResult.inlier_ratio,
      rmse_px: currentResult.rmse_px,
      ground_truth_corner_error_px: currentResult.ground_truth_corner_error_px,
      homography: currentResult.homography,
      correspondence_image: currentResult.images.correspondence_image,
      warped_image: currentResult.images.warped_image,
      overlay_image: currentResult.images.overlay_image,
      source_preprocessed: currentResult.images.source_preprocessed,
      reference_preprocessed: currentResult.images.reference_preprocessed,
      notes: "",
    };
    const res = await Api.saveObservation(payload);
    saveStatus.innerHTML = `✅ Saved as Observation #${res.id} — <a href="observations.html?id=${res.id}">view it</a>`;
  } catch (e) {
    saveStatus.textContent = "❌ Save failed: " + e.message;
  }
});
