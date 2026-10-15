# FoodVision AI

FoodVision is a presentation-ready prototype of the system described in the supplied paper, **“Food Image Classification and Data Extraction Using Convolutional Neural Network and Web Crawlers.”**

The complete product story is:

```text
Food image → CNN prediction → food label → knowledge layer → nutrition insight
```

The project supports a clearly labeled **Model simulator** now and a clean handoff to the team's trained model through **Live model mode** later. The simulator is not image recognition; it exists to exercise the interface and information-retrieval flow while the model team finishes training.

## Run the complete demo

The backend server uses Python's standard library for the presentation flow, so no installation is required for Demo mode:

```bash
python3 backend/server.py
```

Open:

```text
http://localhost:8000
```

The server provides both the web app and API endpoints:

```text
GET  /health
GET  /api/v1/status
POST /api/v1/predict
GET  /api/v1/insights?label=<class_id>&online=0|1
```

If port 8000 is already busy:

```bash
FOODVISION_PORT=8001 python3 backend/server.py
```

Then open `http://localhost:8001`.

## Presentation flow

1. Open **Food scanner**.
2. Select **Live model**. This is the true project path and will return an explicit model-unavailable state until the team model is installed.
3. To explain the completed post-training UI before handoff, select **Model simulator**, choose any class from the dynamic project class list, and click **Run simulated model result**.
4. Or select one of the real uploaded photos in the **Uploaded demo photos** gallery. The selected photo is shown in the preview, then the simulator runs the downstream result flow.
5. Show the ranked predictions, confidence, nutrition estimate, ingredients, origin, recipe, retrieval sources, and health insight. The result is labeled **DEMO** and **illustrative** where appropriate.
6. Use **Download JSON report** if you want to save the current result as a portable JSON report.
7. Open **How it works** and explain image input → preprocessing → CNN classifier → information retrieval → user insight.
8. Show the model card. The current displayed accuracy values are marked as **paper reference** until the team replaces them with measured results.
9. After the model handoff, upload a real image in **Live model** mode. The food label must come from `/api/v1/predict`; the frontend then calls `/api/v1/insights` for enrichment.

The initial screen is intentionally empty. That is the truthful state before a trained model produces a prediction; no food is silently presented as the answer.

## Demo mode and live model mode

### Model simulator

The simulator uses local files under `data/`:

- `data/foods.json` — structured food information
- `data/demo-predictions.json` — optional deterministic ranked predictions for classes that have fixtures
- `data/demo-images.json` — the real uploaded photo gallery and class mapping
- `data/model-metrics.json` — paper reference metrics and a placeholder for the team's measured metrics
- `data/class_names.json` — stable class IDs used by the model contract
- `data/config.json` — input size, thresholds, API path, and upload limit

The simulator does not inspect or classify the uploaded image. It generates a clearly labeled model-output example from the same class IDs the trained model will use, then exercises the real knowledge-layer contract. This keeps the UI honest and prevents six hardcoded food buttons from being mistaken for a classifier.

The uploaded photos are stored in `foods-example/`. They are real visual inputs for the presentation gallery. Selecting a photo previews it, but the simulator still does not claim that the photo was recognized by a trained model.

### Live model mode

The frontend sends an uploaded image to:

```text
POST /api/v1/predict
```

The backend loads a Keras model only when all of the following are true:

1. A model exists at `model/foodvision_model.keras`, or `FOODVISION_MODEL_PATH` points to one.
2. The optional packages in `backend/requirements.txt` are installed.
3. The model output order matches `data/class_names.json`.

Until then, the API returns HTTP 503 with `MODEL_UNAVAILABLE`. The frontend shows an explicit error and does not invent a food result for the uploaded image.

## Team model handoff

Ask the training team for:

1. A `.keras` or `.h5` model file.
2. The exact class-name order used during training.
3. Input image size and preprocessing details.
4. Measured training accuracy, validation accuracy, class count, and inference time.

Copy the model into the project:

```text
model/foodvision_model.keras
```

If the model has a different path:

```bash
FOODVISION_MODEL_PATH=/absolute/path/to/model.keras python3 backend/server.py
```

Install the optional model dependencies in an environment compatible with the TensorFlow version:

```bash
python3 -m pip install -r backend/requirements.txt
```

For a model trained with Inception V3, the expected default preprocessing is:

```text
inception_v3_preprocess_input
```

If the team used a different input size or preprocessing, set:

```bash
FOODVISION_INPUT_SIZE=224 FOODVISION_PREPROCESSING=mobile_net_preprocess_input python3 backend/server.py
```

The API returns this shape:

```json
{
  "status": "success",
  "requestId": "uuid",
  "mode": "live",
  "model": {
    "name": "FoodVision team model",
    "version": "unreleased",
    "inputSize": 299,
    "preprocessing": "inception_v3_preprocess_input",
    "available": true
  },
  "predictions": [
    {
      "classId": "chicken_wings",
      "label": "Chicken wings",
      "confidence": 0.97
    }
  ],
  "latencyMs": 1600
}
```

## What the paper contributes to the project

The supplied paper describes:

- Food-101 image classification
- Inception V3 transfer learning
- Data augmentation using shear, zoom, and horizontal flip
- 20-class and 25-class experiments
- Recipe and nutrition retrieval
- Food origin retrieval
- Nearby restaurant retrieval is represented as a future API/crawler boundary, not invented local results
- Data cleaning and document export

The current prototype implements the complete user-facing flow with a local knowledge layer. Live Scrapy/Selenium crawlers are intentionally not required for the presentation because third-party websites, CAPTCHA, location permissions, and network conditions can make a live demo unreliable. They can be connected later behind a server-side cache or replaced with approved APIs.

## Metrics wording

The displayed paper reference values are:

- Training accuracy: **97.00%** for 20 classes
- Validation accuracy: **92.23%** for 20 classes
- Reference inference time: **1.6 seconds**

Do not call these the team's results until the team has measured its own model. Replace the `deployed` section in `data/model-metrics.json` after evaluation.

## Files

- `index.html` — application layout
- `styles.css` — visual design and responsive states
- `app.js` — demo/live mode, prediction rendering, upload flow, and state machine
- `data/foods.json` — local food knowledge layer
- `data/demo-predictions.json` — optional simulator fixtures
- `data/class_names.json` — model label mapping
- `data/model-metrics.json` — reference and deployed metric records
- `data/config.json` — runtime configuration
- `backend/server.py` — static server and optional Keras prediction API
- `backend/requirements.txt` — optional live-model dependencies
- `PRESENTATION_GUIDE.md` — speaking flow and demo script
- `MODEL_HANDOFF.md` — training-team integration checklist
- `1-s2.0-S1877050922025042-main.pdf` — supplied reference paper
