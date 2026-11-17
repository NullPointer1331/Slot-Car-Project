"""Lap-based TensorFlow training and MicroPython model export."""

import argparse
import json
import os

MODEL_FEATURES = [
    "section_id",
    "distance_m",
    "speed_m_s",
    "accel_x_m_s2",
    "accel_y_m_s2",
    "accel_z_m_s2",
    "gyro_x_dps",
    "gyro_y_dps",
    "gyro_z_dps",
    "requested_throttle",
]

CRASH_HORIZON_S = 0.5
SAFETY_MARGIN = 0.10
POSITION_BIN_M = 0.5
TEST_LAP_FRACTION = 0.20

model = None


def load_model(model_path):
    """Load a TensorFlow model from a Keras or SavedModel path."""
    import tensorflow as tf

    try:
        loaded = tf.keras.models.load_model(model_path)
        print("Model loaded successfully from {}".format(model_path))
        return loaded
    except Exception as error:
        print("Error loading model from {}: {}".format(model_path, error))
        return None


def load_dataset(dataset_path):
    """Load a CSV file as a pandas DataFrame."""
    import pandas as pd

    try:
        data = pd.read_csv(dataset_path)
        print("Dataset loaded successfully from {}".format(dataset_path))
        return data
    except Exception as error:
        print("Error loading dataset from {}: {}".format(dataset_path, error))
        return None


def label_crashes(data, horizon_s=CRASH_HORIZON_S):
    """Mark a row as crashing if a crash occurs during the next horizon."""
    import pandas as pd

    result = data.copy()
    if "lap_id" not in result:
        result["lap_id"] = 1
    if "section_id" not in result:
        result["section_id"] = 0
    if "elapsed_s" not in result:
        raise ValueError("CSV must contain elapsed_s")

    if "crash_event" in result:
        result["_event"] = pd.to_numeric(
            result["crash_event"], errors="coerce"
        ).fillna(0)
    elif "state" in result:
        result["_event"] = result["state"].astype(str).str.upper().eq("CRASH").astype(int)
    else:
        raise ValueError("CSV must contain crash_event or state")

    result["crash"] = 0
    for _, lap in result.groupby("lap_id", sort=False):
        lap = lap.sort_values("elapsed_s")
        event_times = lap.loc[lap["_event"] > 0, "elapsed_s"].tolist()
        for index, time in zip(lap.index, lap["elapsed_s"]):
            result.loc[index, "crash"] = (
                1
                if any(
                    0.0 <= crash_time - time <= horizon_s
                    for crash_time in event_times
                )
                else 0
            )
    return result


def build_safe_targets(data, safety_margin=SAFETY_MARGIN, position_bin_m=POSITION_BIN_M):
    """Build a continuous safe-throttle target at each section/position."""
    import pandas as pd

    result = data.copy()
    if "crash" not in result:
        result = label_crashes(result)
    if "lap_id" not in result:
        result["lap_id"] = 1

    result["_position_bin"] = (
        pd.to_numeric(result["distance_m"], errors="coerce").fillna(0.0) // position_bin_m
    )
    result["_section"] = pd.to_numeric(
        result["section_id"], errors="coerce"
    ).fillna(0).astype(int)

    safe = result[result["crash"] == 0]
    safe_targets = (
        safe.groupby(["_section", "_position_bin"])["requested_throttle"]
        .max()
        .sub(safety_margin)
        .clip(lower=0.0, upper=1.0)
    )
    fallback = safe["requested_throttle"].max() if len(safe) else 0.30
    fallback = max(0.0, min(1.0, fallback - safety_margin))

    result["target_throttle"] = [
        float(safe_targets.get((int(section), int(position)), fallback))
        for section, position in zip(result["_section"], result["_position_bin"])
    ]
    return result


def split_laps(data, test_fraction=TEST_LAP_FRACTION, random_seed=1337):
    """Split rows while keeping complete laps in each partition."""
    import numpy as np

    lap_ids = list(dict.fromkeys(data["lap_id"].tolist()))
    if len(lap_ids) < 2:
        raise ValueError("At least two laps are required for train/test splitting")
    rng = np.random.default_rng(random_seed)
    rng.shuffle(lap_ids)
    test_count = max(1, int(round(len(lap_ids) * test_fraction)))
    test_count = min(test_count, len(lap_ids) - 1)
    test_laps = set(lap_ids[:test_count])
    train_laps = set(lap_ids[test_count:])
    train = data[data["lap_id"].isin(train_laps)].copy()
    test = data[data["lap_id"].isin(test_laps)].copy()
    return train, test


def scaled_features(data, feature_names=MODEL_FEATURES):
    """Convert model features to finite float arrays."""
    import pandas as pd

    missing = [name for name in feature_names if name not in data]
    if missing:
        raise ValueError("Missing feature columns: {}".format(", ".join(missing)))
    values = data[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return values


def train_model(train_data, test_data):
    """Train the 16/16 ReLU regression model with early stopping."""
    import numpy as np
    import tensorflow as tf

    x_train = scaled_features(train_data)
    x_test = scaled_features(test_data)
    means = x_train.mean(axis=0).astype(np.float32)
    scales = x_train.std(axis=0).replace(0.0, 1.0).astype(np.float32)
    x_train_scaled = ((x_train - means) / scales).astype(np.float32).to_numpy()
    x_test_scaled = ((x_test - means) / scales).astype(np.float32).to_numpy()
    y_train = train_data["target_throttle"].astype(np.float32).to_numpy()
    y_test = test_data["target_throttle"].astype(np.float32).to_numpy()

    tf.keras.utils.set_random_seed(1337)
    model_instance = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(MODEL_FEATURES),)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model_instance.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="mse",
        metrics=["mae"],
    )
    history = model_instance.fit(
        x_train_scaled,
        y_train,
        validation_data=(x_test_scaled, y_test),
        epochs=500,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True
            )
        ],
        verbose=0,
    )
    model_instance.evaluate(x_test_scaled, y_test, verbose=0)
    return model_instance, means, scales, history


def section_throttle_caps(train_data, safety_margin=SAFETY_MARGIN):
    """Return highest observed safe throttle by section for offline clamping."""
    safe = train_data[train_data["crash"] == 0]
    caps = {}
    if len(safe):
        for section, throttle in safe.groupby("_section")["requested_throttle"].max().items():
            caps[str(int(section))] = max(0.0, min(1.0, float(throttle) - safety_margin))
    return caps


def export_model(model_instance, means, scales, train_data, output_path):
    """Export weights, scaling, feature order, and section caps as JSON."""
    weights = model_instance.get_weights()
    export = {
        "format": "msr_dense_v1",
        "features": MODEL_FEATURES,
        "mean": [float(value) for value in means],
        "scale": [float(value) for value in scales],
        "layers": [
            {"kernel": weights[0].tolist(), "bias": weights[1].tolist(), "activation": "relu"},
            {"kernel": weights[2].tolist(), "bias": weights[3].tolist(), "activation": "relu"},
            {"kernel": weights[4].tolist(), "bias": weights[5].tolist(), "activation": "sigmoid"},
        ],
        "section_caps": section_throttle_caps(train_data),
    }
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w") as output:
        json.dump(export, output)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default=os.path.join("Training", "data.csv"))
    parser.add_argument("--export", default=os.path.join("Training", "throttle_model.json"))
    parser.add_argument("--keras-output", default=os.path.join("Training", "throttle.keras"))
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    if data is None:
        return 1
    data = build_safe_targets(label_crashes(data))
    train, test = split_laps(data)
    model_instance, means, scales, history = train_model(train, test)
    model_instance.save(args.keras_output)
    export_model(model_instance, means, scales, train, args.export)

    predictions = model_instance.predict(
        ((scaled_features(test) - means) / scales).astype("float32"),
        verbose=0,
    )
    mae = float(abs(predictions[:, 0] - test["target_throttle"].to_numpy()).mean())
    print("Test MAE: {:.4f}".format(mae))
    print("Exported: {}".format(args.export))
    print("Epochs: {}".format(len(history.history["loss"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
