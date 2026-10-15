# Trained model handoff

Copy the team's trained Keras model here as:

```text
foodvision_model.keras
```

The backend will detect it automatically when TensorFlow, NumPy, and Pillow are installed. If the file has another name or location, use:

```bash
FOODVISION_MODEL_PATH=/absolute/path/to/model.keras python3 backend/server.py
```

Keep the model's class-index order synchronized with `data/class_names.json`. See `MODEL_HANDOFF.md` for the complete contract.
