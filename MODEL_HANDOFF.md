# Model handoff checklist

This file is the contract between the model-training team and the FoodVision demo.

## Required handoff

- [ ] Model file: `.keras` or `.h5`
- [ ] Ordered class-label file
- [ ] Image input size
- [ ] Preprocessing function
- [ ] Output activation and class-index convention
- [ ] Training accuracy measured on the training split
- [ ] Validation accuracy measured on a held-out split
- [ ] Test accuracy, if available
- [ ] Measured inference latency
- [ ] Model version/date

## Default project assumptions

The current project assumes:

```text
Model: Inception V3 transfer learning
Input: 299 × 299 RGB image
Preprocessing: inception_v3_preprocess_input
Output: softmax probabilities in class_names.json order
Top-k: 3
```

If any assumption is different, update `data/config.json` or start the server with environment variables.

## Model placement

Preferred path:

```text
model/foodvision_model.keras
```

Alternative:

```bash
FOODVISION_MODEL_PATH=/absolute/path/to/model.keras python3 backend/server.py
```

## Label order is critical

The index of each model output must match the same index in:

```text
data/class_names.json
```

For example, if output index `2` means `chicken_wings` during training, index `2` must remain `chicken_wings` in the handoff file.

## API response expected by the frontend

```json
{
  "status": "success",
  "mode": "live",
  "model": {
    "name": "FoodVision InceptionV3",
    "version": "team-2026-10-14",
    "inputSize": 299,
    "preprocessing": "inception_v3_preprocess_input",
    "available": true
  },
  "predictions": [
    {"classId": "chicken_wings", "label": "Chicken wings", "confidence": 0.97},
    {"classId": "hot_dog", "label": "Hot dog", "confidence": 0.02},
    {"classId": "french_fries", "label": "French fries", "confidence": 0.01}
  ],
  "latencyMs": 1600
}
```

## Replace the metric placeholder

Edit the `deployed` section in `data/model-metrics.json`:

```json
{
  "label": "Team model",
  "status": "evaluated",
  "model": "Inception V3 + transfer learning",
  "classes": 20,
  "epochs": 20,
  "trainingAccuracy": 0.97,
  "validationAccuracy": 0.9223,
  "inferenceTimeSeconds": 1.6,
  "source": "Measured by FoodVision team on held-out validation data"
}
```

Use the team's real measured values, not the sample numbers above.

## Verification before presenting Live mode

```bash
curl -fsS http://localhost:8000/api/v1/status
```

The response must contain:

```json
"mode": "live",
"model": {"available": true}
```

Then test an actual image through the UI and confirm that the result badge says **LIVE MODEL**.
