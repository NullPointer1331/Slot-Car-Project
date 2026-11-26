"""Lap-based TensorFlow training and MicroPython model export."""

import argparse
import json
import math
import os

MODEL_FEATURES = [
    "turn_left",
    "turn_right",
    "grade_uphill",
    "grade_downhill",
    "lap_distance_m",
    "total_distance_m",
    "speed_m_s",
    "accel_x_m_s2",
    "accel_y_m_s2",
    "accel_z_m_s2",
    "gyro_x_dps",
    "gyro_y_dps",
    "gyro_z_dps",
    "previous_throttle",
]

TURN_TYPES = ("straight", "left", "right", "unknown")
GRADE_TYPES = ("level", "uphill", "downhill", "unknown")
NUMERIC_MODEL_FEATURES = [
    "lap_distance_m",
    "total_distance_m",
    "speed_m_s",
    "accel_x_m_s2",
    "accel_y_m_s2",
    "accel_z_m_s2",
    "gyro_x_dps",
    "gyro_y_dps",
    "gyro_z_dps",
    "previous_throttle",
]

CRASH_HORIZON_S = 0.5
SAFETY_MARGIN = 0.10
POSITION_BIN_M = 0.5
TEST_LAP_FRACTION = 0.20
VALIDATION_LAP_FRACTION = 0.20
FAST_LAP_FRACTION = 0.25
FAST_LAP_SAMPLE_WEIGHT = 2.0

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
    """Mark crash samples and preceding samples within the prediction horizon."""
    import numpy as np
    import pandas as pd

    if not np.isfinite(horizon_s) or horizon_s < 0:
        raise ValueError("horizon_s must be nonnegative")
    result = data.copy()
    if "elapsed_s" not in result:
        raise ValueError("CSV must contain elapsed_s")
    elapsed = pd.to_numeric(result["elapsed_s"], errors="coerce")
    if elapsed.isna().any() or not np.isfinite(elapsed).all():
        raise ValueError("elapsed_s must contain finite numeric values")
    result["_elapsed"] = elapsed

    if "crash_event" in result:
        events = pd.to_numeric(result["crash_event"], errors="coerce").fillna(0) > 0
    else:
        events = pd.Series(False, index=result.index)
    if "state" in result:
        events |= result["state"].astype(str).str.upper().eq("CRASH")
    elif "crash_event" not in result:
        raise ValueError("CSV must contain crash_event or state")

    run_ids = result["run_id"] if "run_id" in result else pd.Series(0, index=result.index)
    result["crash"] = 0
    for _, run in result.assign(_event=events, _run_id=run_ids).groupby(
        "_run_id", sort=False
    ):
        run = run.sort_values("_elapsed")
        event_times = run.loc[run["_event"], "_elapsed"].tolist()
        event_index = 0
        labels = []
        for sample_time, event in zip(run["_elapsed"], run["_event"]):
            while (
                event_index < len(event_times)
                and event_times[event_index] < sample_time
            ):
                event_index += 1
            within_horizon = (
                event_index < len(event_times)
                and event_times[event_index] - sample_time <= horizon_s
            )
            labels.append(bool(event) or within_horizon)
        result.loc[run.index, "crash"] = [int(label) for label in labels]
    return result.drop(columns=["_elapsed"])


def build_safe_targets(
    data,
    reference_data=None,
    safety_margin=SAFETY_MARGIN,
    position_bin_m=POSITION_BIN_M,
):
    """Imitate the fastest crash-free training laps with a safety margin."""
    import numpy as np
    import pandas as pd

    if not np.isfinite(safety_margin) or not 0.0 <= safety_margin <= 1.0:
        raise ValueError("safety_margin must be between 0 and 1")
    if not np.isfinite(position_bin_m) or position_bin_m <= 0:
        raise ValueError("position_bin_m must be positive")
    result = data.copy()
    if "crash" not in result:
        result = label_crashes(result)
    if reference_data is None:
        reference = result
    else:
        reference = reference_data.copy()
    elite_laps = fastest_crash_free_laps(reference)
    elite_keys = set(zip(elite_laps["run_id"], elite_laps["lap_id"]))
    reference["_lap_key"] = list(zip(reference["run_id"], reference["lap_id"]))
    result["_lap_key"] = list(zip(result["run_id"], result["lap_id"]))
    for frame in (result, reference):
        frame["_position_bin"] = (
            pd.to_numeric(frame["lap_distance_m"], errors="coerce") // position_bin_m
        )
        frame["_turn_type"] = frame["turn_type"].astype(str).str.lower()
        frame["_grade_type"] = frame["grade_type"].astype(str).str.lower()

    elite_rows = reference[reference["_lap_key"].isin(elite_keys)]
    safe = elite_rows[elite_rows["crash"] == 0]
    safe_targets = (
        safe.groupby(
            ["_turn_type", "_grade_type", "_position_bin"]
        )["requested_throttle"].median()
        .sub(safety_margin)
        .clip(lower=0.0, upper=1.0)
    )
    fallback = safe["requested_throttle"].median() if len(safe) else 0.30
    fallback = max(0.0, min(1.0, fallback - safety_margin))

    result["target_throttle"] = [
        float(safe_targets.get((turn, grade, int(position)), fallback))
        for turn, grade, position in zip(
            result["_turn_type"], result["_grade_type"], result["_position_bin"]
        )
    ]
    return result


def fastest_crash_free_laps(data, fast_fraction=FAST_LAP_FRACTION):
    """Select the fastest fraction of complete crash-free laps."""
    import numpy as np
    import pandas as pd

    if not np.isfinite(fast_fraction) or not 0.0 < fast_fraction <= 1.0:
        raise ValueError("fast_fraction must be greater than 0 and at most 1")
    required = ["run_id", "lap_id", "lap_time_s", "lap_crash_free"]
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError("Missing lap performance columns: {}".format(", ".join(missing)))
    laps = data[required].drop_duplicates(["run_id", "lap_id"])
    laps = laps[laps["lap_crash_free"].astype(bool)].copy()
    if laps.empty:
        raise ValueError("At least one complete crash-free training lap is required")
    laps["lap_time_s"] = pd.to_numeric(laps["lap_time_s"], errors="coerce")
    if laps["lap_time_s"].isna().any() or not np.isfinite(laps["lap_time_s"]).all():
        raise ValueError("Crash-free lap times must be finite numeric values")
    laps = laps.sort_values("lap_time_s", kind="stable")
    elite_count = max(1, int(math.ceil(len(laps) * fast_fraction)))
    return laps.head(elite_count)


def lap_sample_weights(data):
    """Weight laps equally and emphasize examples from the fastest safe laps."""
    import numpy as np

    required = ["run_id", "lap_id"]
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError("Missing lap identity columns: {}".format(", ".join(missing)))
    elite_laps = fastest_crash_free_laps(data)
    elite_keys = set(zip(elite_laps["run_id"], elite_laps["lap_id"]))
    lap_keys = list(zip(data["run_id"], data["lap_id"]))
    counts = data.groupby(required)["lap_id"].transform("size").to_numpy(dtype=float)
    weights = 1.0 / counts
    for index, key in enumerate(lap_keys):
        if key in elite_keys:
            weights[index] *= FAST_LAP_SAMPLE_WEIGHT
    weights /= weights.mean()
    return weights.astype(np.float32)


def split_laps(
    data,
    test_fraction=TEST_LAP_FRACTION,
    validation_fraction=VALIDATION_LAP_FRACTION,
    random_seed=1337,
):
    """Split complete run/lap groups into train, validation, and test partitions."""
    import numpy as np

    if (
        not np.isfinite(test_fraction)
        or not np.isfinite(validation_fraction)
        or test_fraction <= 0
        or validation_fraction <= 0
    ):
        raise ValueError("test and validation fractions must be positive")
    if test_fraction + validation_fraction >= 1:
        raise ValueError("test and validation fractions must sum to less than 1")
    if "lap_id" not in data:
        raise ValueError("Dataset must contain lap_id")
    groups = ["run_id", "lap_id"] if "run_id" in data else ["lap_id"]
    lap_keys = data[groups].drop_duplicates().itertuples(index=False, name=None)
    lap_keys = list(lap_keys)
    if len(lap_keys) < 3:
        raise ValueError("At least three complete laps are required for train/validation/test")
    rng = np.random.default_rng(random_seed)
    rng.shuffle(lap_keys)
    test_count = max(1, int(round(len(lap_keys) * test_fraction)))
    validation_count = max(1, int(round(len(lap_keys) * validation_fraction)))
    if test_count + validation_count >= len(lap_keys):
        test_count = 1
        validation_count = 1

    keys = [tuple(row) for row in data[groups].itertuples(index=False, name=None)]
    test_keys = set(lap_keys[:test_count])
    validation_keys = set(lap_keys[test_count:test_count + validation_count])
    is_test = [key in test_keys for key in keys]
    is_validation = [key in validation_keys for key in keys]
    is_train = [
        not test and not validation
        for test, validation in zip(is_test, is_validation)
    ]
    train = data[is_train].copy()
    validation = data[is_validation].copy()
    test = data[is_test].copy()
    return train, validation, test


def encoded_model_features(data):
    """Create one-hot turn/grade columns plus finite-numeric model candidates."""
    import pandas as pd

    required = NUMERIC_MODEL_FEATURES + ["turn_type", "grade_type"]
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError("Missing feature columns: {}".format(", ".join(missing)))
    turns = data["turn_type"].astype(str).str.lower()
    grades = data["grade_type"].astype(str).str.lower()
    valid_types = turns.isin(TURN_TYPES[:-1]) & grades.isin(GRADE_TYPES[:-1])
    values = pd.DataFrame(
        {
            "turn_left": turns.eq("left").astype(float),
            "turn_right": turns.eq("right").astype(float),
            "grade_uphill": grades.eq("uphill").astype(float),
            "grade_downhill": grades.eq("downhill").astype(float),
        },
        index=data.index,
    )
    values = pd.concat(
        [values, data[NUMERIC_MODEL_FEATURES].apply(pd.to_numeric, errors="coerce")],
        axis=1,
    )[MODEL_FEATURES]
    return values, valid_types


def scaled_features(data, feature_names=MODEL_FEATURES):
    """Convert model features to finite float arrays without hiding bad samples."""
    import numpy as np

    values, valid_types = encoded_model_features(data)
    if not valid_types.all():
        raise ValueError("Model features contain unknown turn or grade types")
    values = values[feature_names]
    invalid = values.isna() | ~np.isfinite(values)
    if invalid.any().any():
        columns = invalid.columns[invalid.any()].tolist()
        raise ValueError("Model features contain missing or non-finite values: {}".format(
            ", ".join(columns)
        ))
    return values


def valid_feature_rows(data):
    """Return rows with a valid IMU and finite values for every model feature."""
    import numpy as np
    import pandas as pd

    missing = [name for name in ["imu_valid"] if name not in data]
    if missing:
        raise ValueError("Missing model-input columns: {}".format(", ".join(missing)))
    features, valid_types = encoded_model_features(data)
    imu_valid = pd.to_numeric(data["imu_valid"], errors="coerce").eq(1)
    valid = imu_valid & valid_types & np.isfinite(features).all(axis=1)
    return data.loc[valid].copy(), int((~valid).sum())


def validate_dataset(data):
    """Reject malformed collection rows and inconsistent landmark-derived lap IDs."""
    import numpy as np
    import pandas as pd

    required = [
        "run_id", "lap_id", "elapsed_s", "landmark_count", "turn_type", "grade_type",
        "lap_distance_m", "requested_throttle", "imu_valid",
    ]
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError("Missing required collection columns: {}".format(", ".join(missing)))
    if data.empty:
        raise ValueError("Dataset contains no samples")
    if data["run_id"].isna().any() or data["run_id"].astype(str).str.strip().eq("").any():
        raise ValueError("run_id must be present for every sample")

    numeric_columns = [
        "lap_id", "elapsed_s", "landmark_count",
        "lap_distance_m", "requested_throttle", "imu_valid",
    ]
    numeric = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError("Collection data contains missing or nonnumeric required values")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Collection data contains non-finite required values")
    if (numeric["elapsed_s"] < 0).any() or (numeric["lap_distance_m"] < 0).any():
        raise ValueError("Elapsed time and lap distance must be nonnegative")
    if not numeric["requested_throttle"].between(0.0, 1.0).all():
        raise ValueError("requested_throttle must be between 0 and 1")
    if not numeric["imu_valid"].isin((0, 1)).all():
        raise ValueError("imu_valid must contain only 0 or 1")
    for column in ("lap_id", "landmark_count"):
        if not (numeric[column] % 1 == 0).all():
            raise ValueError("{} must contain integer values".format(column))
    if not (numeric["lap_id"] == numeric["landmark_count"] // 2).all():
        raise ValueError("lap_id does not match the two-landmark lap convention")
    if not data["turn_type"].astype(str).str.lower().isin(TURN_TYPES).all():
        raise ValueError("turn_type contains an unsupported value")
    if not data["grade_type"].astype(str).str.lower().isin(GRADE_TYPES).all():
        raise ValueError("grade_type contains an unsupported value")
    for _, run in data.assign(_elapsed=numeric["elapsed_s"]).groupby("run_id", sort=False):
        if not run["_elapsed"].is_monotonic_increasing:
            raise ValueError("elapsed_s must be nondecreasing within each run_id")
    return data


def complete_lap_rows(data, lap_metrics=None):
    """Keep complete laps and attach their elapsed-time and crash-free labels."""
    required = ["run_id", "lap_id", "elapsed_s"]
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError("Missing complete-lap columns: {}".format(", ".join(missing)))

    if lap_metrics is None:
        lap_metrics = complete_lap_times(data)
    metric_by_key = {
        (lap["run_id"], lap["lap_id"]): lap for lap in lap_metrics
    }
    keys = list(zip(data["run_id"], data["lap_id"]))
    complete = data.loc[[key in metric_by_key for key in keys]].copy()
    complete["lap_time_s"] = [
        metric_by_key[key]["lap_time_s"]
        for key in zip(complete["run_id"], complete["lap_id"])
    ]
    complete["lap_crash_free"] = [
        metric_by_key[key]["crash_free"]
        for key in zip(complete["run_id"], complete["lap_id"])
    ]
    return complete


def complete_lap_times(data):
    """Return observed durations and crash status for laps between landmarks."""
    import pandas as pd

    required = ["run_id", "lap_id", "elapsed_s"]
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError("Missing lap timing columns: {}".format(", ".join(missing)))
    crash_events = pd.Series(False, index=data.index)
    if "crash_event" in data:
        crash_events |= pd.to_numeric(data["crash_event"], errors="coerce").fillna(0) > 0
    if "state" in data:
        crash_events |= data["state"].astype(str).str.upper().eq("CRASH")

    completed = []
    for run_id, run in data.groupby("run_id", sort=False):
        run = run.sort_values("elapsed_s")
        boundaries = run["lap_id"].ne(run["lap_id"].shift()).to_numpy().nonzero()[0]
        boundaries = boundaries[1:]
        for start, end in zip(boundaries, boundaries[1:]):
            lap_id = int(run["lap_id"].iloc[start])
            next_lap_id = int(run["lap_id"].iloc[end])
            if next_lap_id != lap_id + 1:
                continue
            lap_rows = run.iloc[start:end]
            lap_crashed = bool(crash_events.loc[lap_rows.index].any())
            completed.append(
                {
                    "run_id": run_id,
                    "lap_id": lap_id,
                    "lap_time_s": float(
                        run["elapsed_s"].iloc[end] - run["elapsed_s"].iloc[start]
                    ),
                    "crash_free": not lap_crashed,
                }
            )
    return completed


def train_model(train_data, validation_data, test_data):
    """Train the 16/16 ReLU regression model with early stopping."""
    import numpy as np
    import tensorflow as tf

    x_train = scaled_features(train_data)
    x_validation = scaled_features(validation_data)
    x_test = scaled_features(test_data)
    means = x_train.mean(axis=0).astype(np.float32)
    scales = x_train.std(axis=0).replace(0.0, 1.0).astype(np.float32)
    x_train_scaled = ((x_train - means) / scales).astype(np.float32).to_numpy()
    x_validation_scaled = ((x_validation - means) / scales).astype(np.float32).to_numpy()
    x_test_scaled = ((x_test - means) / scales).astype(np.float32).to_numpy()
    y_train = train_data["target_throttle"].astype(np.float32).to_numpy()
    y_validation = validation_data["target_throttle"].astype(np.float32).to_numpy()
    y_test = test_data["target_throttle"].astype(np.float32).to_numpy()
    train_sample_weights = lap_sample_weights(train_data)

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
        sample_weight=train_sample_weights,
        validation_data=(x_validation_scaled, y_validation),
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
    """Return safe-throttle caps for each observed turn/grade combination."""
    safe = train_data[train_data["crash"] == 0]
    caps = {}
    if len(safe):
        groups = safe.groupby(["_turn_type", "_grade_type"])["requested_throttle"].max()
        for (turn, grade), throttle in groups.items():
            caps["{}|{}".format(turn, grade)] = max(
                0.0, min(1.0, float(throttle) - safety_margin)
            )
    return caps


def export_model(model_instance, means, scales, train_data, output_path):
    """Export weights, scaling, feature order, and section caps as JSON."""
    weights = model_instance.get_weights()
    export = {
        "format": "msr_dense_v2",
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
    data = validate_dataset(data)
    data = label_crashes(data)
    observed_lap_times = complete_lap_times(data)
    data = complete_lap_rows(data, observed_lap_times)
    train_rows, validation_rows, test_rows = split_laps(data)
    partitions = []
    for name, rows in (
        ("training", train_rows),
        ("validation", validation_rows),
        ("test", test_rows),
    ):
        valid_rows, rejected_count = valid_feature_rows(rows)
        print(
            "Excluded invalid IMU/model samples from {}: {}".format(
                name, rejected_count
            )
        )
        if valid_rows.empty:
            raise ValueError("{} partition has no valid model samples".format(name))
        partitions.append(valid_rows)
    train_rows, validation_rows, test_rows = partitions
    fastest_laps = fastest_crash_free_laps(train_rows)
    crash_free_training_laps = train_rows[
        train_rows["lap_crash_free"].astype(bool)
    ][["run_id", "lap_id"]].drop_duplicates()
    print(
        "Lap-time objective: targets from fastest {}/{} crash-free training laps "
        "(cutoff {:.3f} s)".format(
            len(fastest_laps),
            len(crash_free_training_laps),
            float(fastest_laps["lap_time_s"].max()),
        )
    )
    train = build_safe_targets(train_rows)
    validation = build_safe_targets(validation_rows, reference_data=train)
    test = build_safe_targets(test_rows, reference_data=train)
    model_instance, means, scales, history = train_model(train, validation, test)
    model_instance.save(args.keras_output)
    export_model(model_instance, means, scales, train, args.export)

    predictions = model_instance.predict(
        ((scaled_features(test) - means) / scales).astype("float32"),
        verbose=0,
    )
    mae = float(abs(predictions[:, 0] - test["target_throttle"].to_numpy()).mean())
    print("Test MAE: {:.4f}".format(mae))
    crash_free_times = [
        lap["lap_time_s"] for lap in observed_lap_times if lap["crash_free"]
    ]
    if crash_free_times:
        print(
            "Observed complete laps: {}; crash-free: {} (fastest {:.3f} s, mean {:.3f} s)".format(
                len(observed_lap_times),
                len(crash_free_times),
                min(crash_free_times),
                sum(crash_free_times) / len(crash_free_times),
            )
        )
    else:
        print(
            "Observed complete laps: {}; crash-free laps: none".format(
                len(observed_lap_times)
            )
        )
    print("Exported: {}".format(args.export))
    print("Epochs: {}".format(len(history.history["loss"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
