const state = {
  mode: "live",
  phase: "idle",
  foods: {},
  classNames: [],
  config: {},
  demoPredictions: {},
  demoImages: [],
  metrics: {},
  selectedDemoId: "",
  demoPhoto: null,
  file: null,
  objectUrl: null,
  requestId: 0,
  abortController: null,
  dataReady: false,
  lastResult: null,
  modelStatus: null
};

const elements = {};
const byId = (id) => document.getElementById(id);
const camel = (id) => id.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());

async function init() {
  [
    "dropzone", "dropzone-empty", "preview-wrap", "preview-image", "preview-label", "preview-hint", "file-input", "analyze-button", "clear-button",
    "result-empty", "result-empty-title", "result-empty-detail", "processing-state", "processing-title", "processing-detail", "process-image", "process-model", "process-data",
    "result-content", "result-state", "system-status-text", "food-icon", "food-name", "food-description", "confidence-ring", "confidence-value",
    "result-model", "result-input", "result-latency", "top-predictions-list", "metric-calories", "metric-protein", "metric-carbs",
    "metric-fat", "nutrition-note", "health-insight", "food-origin", "food-ingredients", "recipe-steps", "retrieval-status", "overview-source", "recipe-source", "retrieval-note", "source-list", "analysis-source", "new-scan-button", "download-report-button",
    "mode-demo", "mode-live", "stat-classes", "stat-latency", "stat-validation", "model-card-name", "model-card-status",
    "model-training-accuracy", "model-validation-accuracy", "model-card-classes", "model-input-size", "model-card-note", "class-picker", "simulate-button", "connection-title", "photo-gallery", "photo-count"
  ].forEach((id) => { elements[camel(id)] = byId(id); });

  bindNavigation();
  bindUploader();
  bindSamples();
  bindModeSwitch();
  elements.analyzeButton.addEventListener("click", analyzeImage);
  elements.clearButton.addEventListener("click", clearScan);
  elements.newScanButton.addEventListener("click", clearScan);
  elements.downloadReportButton.addEventListener("click", downloadReport);

  try {
    await loadProjectData();
    state.dataReady = true;
    state.mode = state.config.defaultMode || "live";
    state.selectedDemoId = state.classNames[0] || "";
    document.querySelectorAll(".mode-button").forEach((button) => {
      const active = button.dataset.mode === state.mode;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    updateMetricCards();
    populateClassPicker();
    renderPhotoGallery();
    elements.simulateButton.addEventListener("click", simulateModelResult);
    await loadModelStatus();
    clearResultOnly();
    setStatus("Ready for your trained model");
  } catch (error) {
    console.error(error);
    setPhase("error");
    setStatus("Project data could not be loaded");
    showError("Project data is unavailable", "Start the app with `python3 -m http.server 8000` so the data files can load.");
  }
}

async function loadModelStatus() {
  try {
    const response = await fetch(`${state.config.apiBase || "/api/v1"}/status`, { cache: "no-store" });
    if (!response.ok) return;
    const status = await response.json();
    state.modelStatus = status;
    if (status.classSource === "model_handoff" && Array.isArray(status.classNames) && status.classNames.length) {
      state.classNames = status.classNames;
      populateClassPicker();
    }
    const model = status.model || {};
    if (status.modelAvailable) {
      elements.connectionTitle.textContent = `Live model connected · ${model.name || "team model"}`;
      elements.modelCardStatus.textContent = "DEPLOYED";
      elements.modelCardName.textContent = model.name || "Team model";
      elements.modelInputSize.textContent = `${model.inputSize || state.config.inputSize || 299} × ${model.inputSize || state.config.inputSize || 299}`;
    } else {
      elements.connectionTitle.textContent = "Waiting for team model handoff";
    }
    updateMetricCards();
  } catch {
    elements.connectionTitle.textContent = "Live API not reachable";
  }
}

async function loadProjectData() {
  const [foods, classNames, config, demo, demoImages, metrics] = await Promise.all([
    loadJson("data/foods.json"),
    loadJson("data/class_names.json"),
    loadJson("data/config.json"),
    loadJson("data/demo-predictions.json"),
    loadJson("data/demo-images.json"),
    loadJson("data/model-metrics.json")
  ]);
  state.foods = foods.foods || {};
  state.classNames = classNames;
  state.config = config;
  state.demoPredictions = demo.predictions || {};
  state.demoImages = demoImages.images || [];
  state.metrics = metrics;
}

async function loadJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`Could not load ${path} (${response.status})`);
  return response.json();
}

function bindNavigation() {
  document.querySelectorAll(".nav-item").forEach((button) => button.addEventListener("click", () => {
    const view = button.dataset.view;
    document.querySelectorAll(".nav-item").forEach((item) => item.classList.toggle("active", item === button));
    document.querySelectorAll(".nav-item").forEach((item) => item.setAttribute("aria-current", item === button ? "page" : "false"));
    document.querySelectorAll(".view").forEach((section) => section.classList.toggle("active-view", section.id === `${view}-view`));
  }));
  document.querySelectorAll(".nav-item").forEach((item) => item.setAttribute("aria-current", item.classList.contains("active") ? "page" : "false"));
}

function bindUploader() {
  elements.fileInput.addEventListener("change", (event) => handleFile(event.target.files[0]));
  ["dragenter", "dragover"].forEach((eventName) => elements.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropzone.classList.add("drag-over");
  }));
  ["dragleave", "drop"].forEach((eventName) => elements.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    elements.dropzone.classList.remove("drag-over");
  }));
  elements.dropzone.addEventListener("drop", (event) => handleFile(event.dataTransfer.files[0]));
  elements.dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      elements.fileInput.click();
    }
  });
}

function bindSamples() {
  // The model-output selector is populated from data/class_names.json after startup.
}

function bindModeSwitch() {
  document.querySelectorAll(".mode-button").forEach((button) => button.addEventListener("click", () => {
    if (button.dataset.mode === state.mode) return;
    invalidateActiveRequest();
    state.mode = button.dataset.mode;
    document.querySelectorAll(".mode-button").forEach((item) => {
      const active = item === button;
      item.classList.toggle("active", active);
      item.setAttribute("aria-pressed", String(active));
    });
    clearResultOnly();
    if (state.mode === "live") {
      if (state.demoPhoto) clearUploadedImage();
      setStatus("Live model mode selected");
      elements.connectionTitle.textContent = "Live model endpoint selected";
      setPhase(state.file ? "image-selected" : "idle");
      setResultState("LIVE READY");
    } else {
      elements.connectionTitle.textContent = "Model simulator selected";
      setStatus("Model simulator ready");
      setResultState("DEMO");
    }
    updateEmptyState();
  }));
}

function populateClassPicker() {
  elements.classPicker.replaceChildren();
  const simulatorIds = [...new Set([...state.demoImages.map((image) => image.classId), ...state.classNames])];
  simulatorIds.forEach((classId) => {
    const option = document.createElement("option");
    option.value = classId;
    option.textContent = state.foods[classId]?.displayName || classId.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
    elements.classPicker.appendChild(option);
  });
  elements.classPicker.value = state.selectedDemoId;
  elements.simulateButton.disabled = state.classNames.length === 0;
  elements.connectionTitle.textContent = state.mode === "live" ? "Waiting for team model handoff" : "Model simulator selected";
}

function renderPhotoGallery() {
  elements.photoGallery.replaceChildren();
  elements.photoCount.textContent = `${state.demoImages.length} photos`;
  state.demoImages.forEach((photo) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "photo-card";
    button.dataset.classId = photo.classId;
    button.setAttribute("aria-label", `Use ${photo.label} photo in simulator`);
    const image = document.createElement("img");
    image.src = photo.path;
    image.alt = photo.alt || photo.label;
    image.loading = "lazy";
    const meta = document.createElement("span");
    meta.className = "photo-card-meta";
    meta.append(makeTextElement("strong", photo.label), makeTextElement("small", "Use in simulator →"));
    button.append(image, meta);
    button.addEventListener("click", () => selectDemoPhoto(photo));
    elements.photoGallery.appendChild(button);
  });
}

function selectDemoPhoto(photo) {
  invalidateActiveRequest();
  state.mode = "demo";
  state.selectedDemoId = photo.classId;
  elements.modeDemo.classList.add("active");
  elements.modeLive.classList.remove("active");
  elements.modeDemo.setAttribute("aria-pressed", "true");
  elements.modeLive.setAttribute("aria-pressed", "false");
  elements.classPicker.value = photo.classId;
  clearUploadedImage();
  state.demoPhoto = photo;
  elements.previewImage.src = photo.path;
  elements.previewLabel.textContent = "Uploaded photo selected";
  elements.previewHint.textContent = "Simulator input only";
  elements.dropzoneEmpty.classList.add("hidden");
  elements.previewWrap.classList.remove("hidden");
  elements.analyzeButton.disabled = false;
  elements.analyzeButton.querySelector("span:nth-child(2)").textContent = "Analyze selected photo";
  clearResultOnly();
  setStatus(`${photo.label} photo selected for simulator`);
  document.querySelectorAll(".photo-card").forEach((card) => card.classList.toggle("selected", card.dataset.classId === photo.classId));
}

function handleFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    setStatus("Please choose an image file");
    showError("Unsupported file", "Choose a JPG, PNG, or WEBP image.");
    return;
  }
  if (file.size > (state.config.maxUploadBytes || 10485760)) {
    setStatus("Image is larger than 10 MB");
    showError("Image is too large", "Choose an image smaller than 10 MB.");
    return;
  }
  invalidateActiveRequest();
  clearResultOnly();
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.file = file;
  state.demoPhoto = null;
  state.objectUrl = URL.createObjectURL(file);
  elements.previewImage.src = state.objectUrl;
  elements.previewLabel.textContent = "Image ready";
  elements.previewHint.textContent = "Click to replace";
  elements.dropzoneEmpty.classList.add("hidden");
  elements.previewWrap.classList.remove("hidden");
  elements.analyzeButton.disabled = false;
  setPhase("image-selected");
  setStatus(state.mode === "live" ? "Image ready for live model" : "Image ready; choose a model output in the simulator");
}

async function analyzeImage() {
  if ((!state.file && !state.demoPhoto) || !state.dataReady || state.phase === "predicting" || state.phase === "loading-knowledge") return;
  const requestId = ++state.requestId;
  if (state.abortController) state.abortController.abort();
  state.abortController = new AbortController();
  setBusy(true);
  setPhase("predicting");
  setProcessing("Running food classifier", "The image is being passed through the selected prediction path.", 2);
  setStatus(state.mode === "live" ? "Sending image to live model" : "Preparing demo prediction");

  try {
    if (state.mode === "demo") {
      await pause(450);
      if (requestId !== state.requestId) return;
      await renderPrediction(buildSimulationPayload(state.selectedDemoId), "DEMO", requestId);
      setStatus("Model simulation complete");
    } else {
      if (!state.file) throw new Error("Select an image file before using Live model mode.");
      const payload = await requestLivePrediction(state.file, state.abortController.signal);
      if (requestId !== state.requestId) return;
      if (payload.mode === "demo-fallback") {
        renderUnknownPrediction(payload, "LIVE FALLBACK");
        setPhase("fallback");
        setStatus("Live model unavailable; no result was invented");
      } else {
        await renderPrediction(payload, "LIVE MODEL", requestId);
        setStatus("Live model prediction complete");
      }
    }
  } catch (error) {
    if (error.name === "AbortError") return;
    console.error(error);
    setPhase("error");
    showError("Prediction could not be completed", error.message || "Start the backend or switch to Demo mode.");
    setStatus("Prediction error");
  } finally {
    if (requestId === state.requestId) {
      setBusy(false);
      state.abortController = null;
    }
  }
}

async function requestLivePrediction(file, signal) {
  const formData = new FormData();
  formData.append("image", file);
  formData.append("topK", String(state.config.topK || 3));
  const started = performance.now();
  const response = await fetch(`${state.config.apiBase || "/api/v1"}/predict`, { method: "POST", body: formData, signal });
  let payload;
  try { payload = await response.json(); } catch { throw new Error(`Prediction service returned HTTP ${response.status}`); }
  if (!response.ok) throw new Error(payload.message || payload.error || `Prediction service returned HTTP ${response.status}`);
  payload.latencyMs = payload.latencyMs || Math.round(performance.now() - started);
  return payload;
}

function buildSimulationPayload(classId) {
  const selectedId = classId || state.classNames[0];
  const fallbackIds = state.classNames.filter((id) => id !== selectedId).slice(0, 2);
  const predictions = state.demoPredictions[selectedId] || [
    { classId: selectedId, confidence: 0.91 },
    ...fallbackIds.map((id, index) => ({ classId: id, confidence: index === 0 ? 0.06 : 0.03 }))
  ];
  return {
    mode: "demo",
    model: { name: "Presentation simulator (not trained model)", version: "simulator-1", inputSize: state.config.inputSize || 299 },
    predictions: predictions.map((prediction) => ({ ...prediction, label: state.foods[prediction.classId]?.displayName || prediction.classId })),
    latencyMs: 1600,
    food: state.foods[selectedId] || null
  };
}

function simulateModelResult() {
  if (!state.dataReady || state.phase === "predicting" || state.phase === "loading-knowledge") return;
  state.selectedDemoId = elements.classPicker.value;
  state.mode = "demo";
  clearUploadedImage();
  elements.modeDemo.classList.add("active");
  elements.modeLive.classList.remove("active");
  elements.modeDemo.setAttribute("aria-pressed", "true");
  elements.modeLive.setAttribute("aria-pressed", "false");
  setBusy(true);
  setPhase("predicting");
  setProcessing("Replaying the project pipeline", "This simulation demonstrates the interface after a model prediction.", 1);
  void renderPrediction(buildSimulationPayload(state.selectedDemoId), "DEMO", state.requestId).finally(() => {
    setBusy(false);
    setStatus("Model simulation complete");
  });
}

async function renderPrediction(payload, stateLabel, requestId = state.requestId) {
  const predictions = normalizePredictions(payload.predictions);
  const best = predictions[0];
  const food = state.foods[best?.classId] || payload.food || null;
  if (!best || !food || best.confidence < (state.config.confidenceThreshold || 0.5)) {
    renderUnknownPrediction(payload, "UNKNOWN");
    return;
  }
  setPhase("loading-knowledge");
  setProcessing("Retrieving food information", "The predicted class is being passed to the knowledge layer.", 3);
  const insights = await requestInsights(best.classId, stateLabel !== "DEMO");
  if (requestId !== state.requestId) return;
  const lowConfidence = best.confidence < (state.config.lowConfidenceThreshold || 0.75);
  setPhase(lowConfidence ? "low-confidence" : "success");
  elements.resultEmpty.classList.add("hidden");
  elements.processingState.classList.add("hidden");
  elements.resultContent.classList.remove("hidden");
  elements.resultContent.setAttribute("aria-busy", "false");
  setResultState(stateLabel);
  elements.foodIcon.textContent = food.icon;
  elements.foodName.textContent = food.displayName;
  elements.foodDescription.textContent = food.description;
  elements.confidenceValue.textContent = Math.round(best.confidence * 100);
  elements.confidenceRing.style.setProperty("--confidence", `${Math.round(best.confidence * 360)}deg`);
  elements.resultModel.textContent = payload.model?.name || "FoodVision model";
  elements.resultInput.textContent = `${payload.model?.inputSize || state.config.inputSize || 299} × ${payload.model?.inputSize || state.config.inputSize || 299} px`;
  elements.resultLatency.textContent = payload.latencyMs ? `${formatLatency(payload.latencyMs)}` : "Measured on request";
  renderInsights(insights, food);
  renderTopPredictions(predictions);
  if (lowConfidence) {
    setResultState("LOW CONFIDENCE");
    elements.foodDescription.textContent = `Possible match — ${food.description} Review the ranked alternatives before treating this as confirmed.`;
  }
  elements.analysisSource.textContent = stateLabel === "DEMO" ? "Model simulator output · image was not classified" : `${payload.model?.name || "Team model"} · live API response`;
  state.lastResult = { stateLabel, payload, food, predictions, insights };
}

async function requestInsights(classId, online) {
  try {
    const response = await fetch(`${state.config.apiBase || "/api/v1"}/insights?label=${encodeURIComponent(classId)}&online=${online ? "1" : "0"}`, { cache: "no-store" });
    if (!response.ok) throw new Error("Knowledge layer unavailable");
    return await response.json();
  } catch (error) {
    const food = state.foods[classId];
    return food ? {
      classId,
      label: food.displayName,
      overview: { text: food.origin, sourceLabel: "Local illustrative record", status: "local_illustrative" },
      recipe: { title: `${food.displayName} recipe suggestion`, ingredients: String(food.ingredients).split(",").map((item) => item.trim()), steps: food.steps, sourceLabel: "Local illustrative record", status: "local_illustrative", note: "Recipe suggestion; not guaranteed to exactly match the predicted class." },
      nutrition: { status: "illustrative", values: food.nutrition, basis: "Local illustrative estimate; actual nutrition varies by recipe and portion size.", sourceLabel: "Local illustrative record" },
      sources: [{ label: "Local illustrative record", url: null, status: "illustrative" }],
      retrieval: { mode: "offline", status: "local" },
      restaurants: { status: "search_only", note: "Restaurant search is not connected." }
    } : { classId, label: classId, overview: { text: "No verified overview available.", status: "unavailable" }, recipe: { ingredients: [], steps: [], status: "unavailable" }, nutrition: { status: "unavailable" }, sources: [], retrieval: { mode: "offline", status: "unavailable" }, restaurants: { status: "search_only", note: "Restaurant search is not connected." } };
  }
}

function renderInsights(insights, food) {
  const nutrition = insights.nutrition?.values || {};
  elements.metricCalories.textContent = nutrition.calories ?? "—";
  elements.metricProtein.textContent = nutrition.protein == null ? "—" : `${nutrition.protein}g`;
  elements.metricCarbs.textContent = nutrition.carbs == null ? "—" : `${nutrition.carbs}g`;
  elements.metricFat.textContent = nutrition.fat == null ? "—" : `${nutrition.fat}g`;
  elements.nutritionNote.textContent = nutrition.status === "illustrative" ? "Illustrative values · representative serving" : "No verified nutrition returned";
  elements.foodOrigin.textContent = insights.overview?.text || "No verified overview available.";
  elements.foodIngredients.textContent = insights.recipe?.ingredients?.join(", ") || "No verified ingredients available.";
  const recipe = insights.recipe || {};
  renderRecipe(recipe.steps || []);
  const insightText = nutrition.status === "illustrative" ? `${nutrition.basis || "Illustrative estimate only."} ${food.insight.text}` : "Nutrition is unavailable until this class is mapped to a verified food data source.";
  elements.healthInsight.replaceChildren(makeTextElement("span", "ⓘ"), makeInsight(food.insight.title, insightText));
  elements.retrievalStatus.textContent = insights.retrieval?.status || "Unavailable";
  elements.overviewSource.textContent = insights.overview?.sourceLabel || "Unavailable";
  elements.recipeSource.textContent = recipe.sourceLabel || "Unavailable";
  elements.retrievalNote.textContent = nutrition.status === "illustrative" ? "Nutrition is an illustrative estimate, not a measured value for this image or portion." : "No nutrition value is shown because the class has no verified data record.";
  elements.sourceList.replaceChildren();
  (insights.sources || []).forEach((source) => {
    const chip = makeTextElement("span", `${source.label} · ${source.status}`);
    chip.className = `source-chip ${source.status}`;
    elements.sourceList.appendChild(chip);
  });
}

function renderUnknownPrediction(payload, stateLabel) {
  const best = normalizePredictions(payload.predictions)[0];
  setPhase(stateLabel === "LIVE FALLBACK" ? "fallback" : "unknown");
  elements.resultEmpty.classList.add("hidden");
  elements.processingState.classList.add("hidden");
  elements.resultContent.classList.remove("hidden");
  elements.resultContent.setAttribute("aria-busy", "false");
  setResultState(stateLabel);
  elements.foodIcon.textContent = "🔎";
  elements.foodName.textContent = best?.label || "Unknown food item";
  elements.foodDescription.textContent = stateLabel === "LIVE FALLBACK" ? "The trained model is not connected yet, so the system did not invent a result for this uploaded image." : "The model returned a class that is not yet mapped to the project food knowledge base.";
  elements.confidenceValue.textContent = best ? Math.round(best.confidence * 100) : "—";
  elements.confidenceRing.style.setProperty("--confidence", `${best ? Math.round(best.confidence * 360) : 0}deg`);
  elements.resultModel.textContent = payload.model?.name || "Prediction service unavailable";
  elements.resultInput.textContent = `${payload.model?.inputSize || state.config.inputSize || 299} × ${payload.model?.inputSize || state.config.inputSize || 299} px`;
  elements.resultLatency.textContent = payload.latencyMs ? formatLatency(payload.latencyMs) : "Not measured";
  elements.metricCalories.textContent = "—";
  elements.metricProtein.textContent = "—";
  elements.metricCarbs.textContent = "—";
  elements.metricFat.textContent = "—";
  elements.nutritionNote.textContent = "Nutrition withheld until the class is verified";
  elements.foodOrigin.textContent = "No verified food record is available for this prediction yet.";
  elements.foodIngredients.textContent = "Connect the trained class label to data/foods.json before showing nutrition values.";
  clearRetrievalEvidence();
  elements.healthInsight.replaceChildren(makeTextElement("span", "ⓘ"), makeInsight("Knowledge match needed", "The system does not guess nutrition values when the food class is unverified."));
  elements.recipeSteps.replaceChildren(makeRecipeStep("!", "Connect the trained model and map its class output to the knowledge layer."));
  renderTopPredictions(normalizePredictions(payload.predictions));
  elements.analysisSource.textContent = stateLabel === "LIVE FALLBACK" ? "Live API unavailable · explicit fallback" : "Prediction label not mapped to local food records";
  state.lastResult = { stateLabel, payload, food: null, predictions: normalizePredictions(payload.predictions) };
}

function normalizePredictions(predictions) {
  return (predictions || []).map((prediction) => ({
    classId: prediction.classId || prediction.label || "unknown",
    label: prediction.label || state.foods[prediction.classId]?.displayName || prediction.classId || "Unknown",
    confidence: Number(prediction.confidence || 0)
  })).sort((a, b) => b.confidence - a.confidence).slice(0, state.config.topK || 3);
}

function renderTopPredictions(predictions) {
  elements.topPredictionsList.replaceChildren();
  predictions.forEach((prediction, index) => {
    const row = document.createElement("div");
    row.className = "top-prediction";
    const rank = makeTextElement("span", String(index + 1)); rank.className = "top-prediction-rank";
    const label = makeTextElement("span", prediction.label); label.className = "top-prediction-label";
    const track = document.createElement("div"); track.className = "prediction-track";
    const fill = document.createElement("span"); fill.style.width = `${Math.max(2, Math.round(prediction.confidence * 100))}%`; track.appendChild(fill);
    const percent = makeTextElement("span", `${Math.round(prediction.confidence * 100)}%`); percent.className = "top-prediction-percent";
    row.append(rank, label, track, percent);
    elements.topPredictionsList.appendChild(row);
  });
}

function renderRecipe(steps) {
  elements.recipeSteps.replaceChildren();
  (steps || []).forEach((step, index) => elements.recipeSteps.appendChild(makeRecipeStep(String(index + 1), step)));
}

function makeRecipeStep(marker, text) {
  const item = document.createElement("div"); item.className = "recipe-step";
  const badge = makeTextElement("b", marker); const copy = makeTextElement("span", text);
  item.append(badge, copy);
  return item;
}

function makeTextElement(tag, text) {
  const element = document.createElement(tag); element.textContent = text; return element;
}

function makeInsight(title, text) {
  const wrapper = document.createElement("div");
  wrapper.append(makeTextElement("strong", title), makeTextElement("p", text));
  return wrapper;
}

function updateMetricCards() {
  const reference = state.metrics.reference || {};
  const deployed = state.modelStatus?.deployedMetrics || state.metrics.deployed || {};
  const useDeployed = Boolean(state.modelStatus?.modelAvailable && deployed.status === "evaluated");
  const displayed = useDeployed ? deployed : reference;
  const classes = state.classNames.length || reference.classes || 20;
  elements.statClasses.textContent = classes;
  elements.statLatency.textContent = displayed.inferenceTimeSeconds ? `${displayed.inferenceTimeSeconds}s` : "—";
  elements.statValidation.textContent = formatPercent(displayed.validationAccuracy);
  elements.modelCardName.textContent = displayed.model || "Transfer-learning classifier";
  elements.modelCardStatus.textContent = useDeployed ? "DEPLOYED" : "REFERENCE";
  elements.modelTrainingAccuracy.textContent = formatPercent(displayed.trainingAccuracy);
  elements.modelValidationAccuracy.textContent = formatPercent(displayed.validationAccuracy);
  elements.modelCardClasses.textContent = displayed.classes || classes;
  elements.modelInputSize.textContent = `${state.config.inputSize || 299} × ${state.config.inputSize || 299}`;
  elements.modelCardNote.textContent = useDeployed ? `${displayed.source || "Team measured model"}. These are deployed-model metrics.` : `${reference.source || "Reference experiment"}. Replace these values with your team's measured metrics after training.`;
}

function formatPercent(value) { return Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : "Pending"; }
function formatLatency(value) { return value >= 1000 ? `${(value / 1000).toFixed(2)}s` : `${Math.round(value)}ms`; }

function setProcessing(title, detail, activeStep) {
  elements.resultEmpty.classList.add("hidden");
  elements.resultContent.classList.add("hidden");
  elements.resultContent.setAttribute("aria-busy", "true");
  elements.processingState.classList.remove("hidden");
  elements.processingTitle.textContent = title;
  elements.processingDetail.textContent = detail;
  [elements.processImage, elements.processModel, elements.processData].forEach((step, index) => step.classList.toggle("active", index < activeStep));
}

function setPhase(phase) { state.phase = phase; }

function setBusy(isBusy) {
  elements.analyzeButton.disabled = isBusy || (!state.file && !state.demoPhoto);
  elements.clearButton.disabled = isBusy;
  document.querySelectorAll(".mode-button").forEach((button) => { button.disabled = isBusy; });
  elements.classPicker.disabled = isBusy;
  elements.simulateButton.disabled = isBusy;
  elements.analyzeButton.querySelector("span:nth-child(2)").textContent = isBusy ? "Analyzing..." : (state.demoPhoto ? "Analyze selected photo" : "Analyze image");
}

function setStatus(message) { elements.systemStatusText.textContent = message; }

function setResultState(label) {
  const stateClass = String(label).toLowerCase().replace(/[^a-z]+/g, "-").replace(/^-|-$/g, "");
  elements.resultState.textContent = label;
  elements.resultState.className = `result-state ${stateClass}`;
}

function showError(title, detail) {
  elements.resultEmpty.classList.add("hidden");
  elements.processingState.classList.add("hidden");
  elements.resultContent.classList.remove("hidden");
  elements.resultContent.setAttribute("aria-busy", "false");
  setResultState("ERROR");
  elements.foodIcon.textContent = "!";
  elements.foodName.textContent = title;
  elements.foodDescription.textContent = detail;
  elements.confidenceValue.textContent = "—";
  elements.confidenceRing.style.setProperty("--confidence", "0deg");
  elements.resultModel.textContent = "Not available";
  elements.resultInput.textContent = "—";
  elements.resultLatency.textContent = "—";
  [elements.metricCalories, elements.metricProtein, elements.metricCarbs, elements.metricFat].forEach((element) => { element.textContent = "—"; });
  elements.nutritionNote.textContent = "No nutrition returned";
  elements.foodOrigin.textContent = "Start the live backend after the team model handoff, or use the model simulator to explain the post-training flow.";
  elements.foodIngredients.textContent = "No data was fabricated for this failed request.";
  clearRetrievalEvidence();
  elements.healthInsight.replaceChildren(makeTextElement("span", "ⓘ"), makeInsight("Recoverable error", detail));
  elements.recipeSteps.replaceChildren(makeRecipeStep("!", "Upload an image, connect the model, or run the clearly labeled simulator."));
  elements.topPredictionsList.replaceChildren();
  elements.analysisSource.textContent = "System message · no prediction claimed";
  state.lastResult = { stateLabel: "ERROR", payload: {}, food: null, predictions: [] };
}

function clearRetrievalEvidence() {
  elements.retrievalStatus.textContent = "Not run";
  elements.overviewSource.textContent = "Not run";
  elements.recipeSource.textContent = "Not run";
  elements.retrievalNote.textContent = "The predicted class will be sent to the knowledge layer. Nutrition is shown only when a labeled record is available.";
  elements.sourceList.replaceChildren();
}

function downloadReport() {
  if (!state.lastResult) return;
  const report = {
    generatedAt: new Date().toISOString(),
    project: "FoodVision AI",
    mode: state.lastResult.stateLabel,
    model: state.lastResult.payload.model || null,
    latencyMs: state.lastResult.payload.latencyMs || null,
    predictions: state.lastResult.predictions,
    food: state.lastResult.food,
    insights: state.lastResult.insights || null,
    note: "Nutrition values are estimated representative-serving values and are not medical advice."
  };
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `foodvision-report-${state.lastResult.food?.classId || "analysis"}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

function clearResultOnly() {
  elements.resultContent.classList.add("hidden");
  elements.processingState.classList.add("hidden");
  elements.resultEmpty.classList.remove("hidden");
  setResultState(state.mode === "live" ? "LIVE READY" : "DEMO");
  updateEmptyState();
}

function updateEmptyState() {
  if (state.mode === "demo") {
    elements.resultEmptyTitle.textContent = "Ready to simulate the model handoff";
    elements.resultEmptyDetail.textContent = "Choose a class from the simulator and run the post-training result flow. No image classification is claimed.";
  } else {
    elements.resultEmptyTitle.textContent = "Ready for a model prediction";
    elements.resultEmptyDetail.textContent = "Upload an image. The trained model will produce the food label, then the information layer will enrich it.";
  }
}

function invalidateActiveRequest() {
  state.requestId += 1;
  if (state.abortController) state.abortController.abort();
  state.abortController = null;
}

function clearUploadedImage() {
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = null;
  state.file = null;
  state.demoPhoto = null;
  elements.fileInput.value = "";
  elements.previewImage.removeAttribute("src");
  elements.previewLabel.textContent = "Image ready";
  elements.previewHint.textContent = "Click to replace";
  elements.previewWrap.classList.add("hidden");
  elements.dropzoneEmpty.classList.remove("hidden");
  elements.analyzeButton.disabled = true;
  elements.analyzeButton.querySelector("span:nth-child(2)").textContent = "Analyze image";
}

function clearScan() {
  invalidateActiveRequest();
  clearUploadedImage();
  clearResultOnly();
  setPhase("idle");
  setBusy(false);
  setStatus(state.mode === "live" ? "Live model mode selected" : "Demo engine ready");
}

function pause(milliseconds) { return new Promise((resolve) => setTimeout(resolve, milliseconds)); }

document.addEventListener("DOMContentLoaded", init);
