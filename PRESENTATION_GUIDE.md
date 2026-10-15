# FoodVision presentation guide

## One-minute project explanation

> FoodVision is an AI food image classification and information retrieval system. A user uploads a food image, and a transfer-learning CNN predicts the food category. The predicted label is then connected to a structured knowledge layer containing estimated nutrition, ingredients, recipe steps, origin, and health guidance. The current prototype has a model simulator for explaining the post-training interface, while Live model mode accepts the team's trained model through a prediction API.

## Recommended live demonstration

### Step 1 — Introduce the problem

Say:

> People can see a food image, but they may not know the food's name, nutrition, ingredients, or preparation. Our system combines computer vision with food information retrieval to turn an image into useful insight.

### Step 2 — Show the scanner

Open `http://localhost:8000`. The initial screen is intentionally empty because no trained model result should be invented.

Select **Model simulator**, choose any class from the dynamic model-output selector, and click **Run simulated model result**. The selector is generated from the class-label file; it is not a set of classifier buttons.

You can also choose a real image from the **Uploaded demo photos** gallery. For example, select the Chole bhature photo to make the presentation visually match the project’s intended food-recognition use case. Explain that the simulator demonstrates the downstream interface until the trained CNN is connected.

Say:

> This is the model handoff simulator. It does not claim that the uploaded image was classified. It lets us demonstrate the real downstream interface using the same class-label contract that the trained CNN will return.

### Step 3 — Explain the result

Point to:

- Detected food
- Confidence
- Ranked predictions
- Calories, protein, carbohydrates, and fat
- Origin and key ingredients
- Retrieval source evidence
- Quick preparation
- Health insight

The **Download report** button exports the current prediction and food record as a portable JSON report. It is a deterministic local equivalent of the paper's data-export step.

Say:

> The classifier produces a ranked prediction. The winning class becomes a key used by the knowledge layer. The application does not calculate nutrition from pixels; it retrieves an estimated food record for the predicted class.

### Step 4 — Show the workflow

Open **How it works**.

Explain:

```text
Image input
  → Preprocessing
  → Inception V3 transfer-learning classifier
  → Predicted food label
  → Nutrition/recipe/origin knowledge layer
  → User dashboard
```

### Step 5 — Explain the model results

Point to the model card.

Say:

> The values shown as paper reference come from the supplied research paper: 97% training accuracy and 92.23% validation accuracy for 20 classes. Once our team model is evaluated, we will replace these with our measured results and label them as deployed-model metrics.

### Step 6 — Explain future integration

Say:

> The paper describes Scrapy and Selenium crawlers for recipe, nutrition, origin, and restaurant information. We kept the presentation layer local and deterministic so the demo does not depend on third-party website availability. In production, this knowledge layer can use approved APIs or cached crawlers.

## If the team model is ready

1. Copy the model to `model/foodvision_model.keras`.
2. Update `data/class_names.json` if the trained class order differs.
3. Update the deployed values in `data/model-metrics.json`.
4. Start `python3 backend/server.py`.
5. Select **Live model** in the top-right switch.
6. Upload an image and click **Analyze image**.

The result should show **LIVE MODEL**, the model name, actual latency, and the ranked model predictions.

## If the team model is not ready

Do not pretend it is live. Use Model simulator mode and say:

> The trained model is being finalized by the model team. This interface already defines the complete inference-to-insight contract. The simulator is explicitly labeled and demonstrates the exact downstream flow that will run after the model is connected.

## Questions and answers

### Why transfer learning?

> A pre-trained model already knows general visual features such as edges, textures, and shapes. Fine-tuning it for food classes reduces training time and computation compared with training a CNN from zero.

### Why use a local knowledge base in the demo?

> Nutrition and recipe values need a verified source and can be slow or unreliable when scraped live. A structured local layer makes the demonstration reproducible. Live APIs or cached crawlers are the production extension.

### Does the image itself give exact calories?

> No. The image predicts the food identity. Nutrition is an estimate based on a representative serving and depends on portion size, ingredients, and cooking method.

### What happens with a low-confidence prediction?

> The system should show the top alternatives or request confirmation instead of presenting an uncertain label as fact. This is why the interface includes ranked predictions and an unknown state.

### What is the next technical step?

> Connect the team's evaluated Food-101/Inception V3 model through `/api/v1/predict`, verify preprocessing and label order, and replace the paper reference metrics with measured project metrics.
