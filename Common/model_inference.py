"""Small JSON dense-network inference for MicroPython."""

import json


class ThrottleModel:
    """Run the exported 16/16 ReLU throttle model without NumPy or TensorFlow."""

    def __init__(self, model_path):
        with open(model_path, "r") as model_file:
            model = json.load(model_file)
        if model.get("format") != "msr_dense_v1":
            raise ValueError("Unsupported throttle model format")
        self.features = model["features"]
        self.mean = model["mean"]
        self.scale = model["scale"]
        self.layers = model["layers"]
        self.section_caps = model.get("section_caps", {})

    @staticmethod
    def _relu(value):
        return value if value > 0.0 else 0.0

    def predict(self, feature_values):
        """Predict one normalized throttle in the 0-1 range."""
        if len(feature_values) != len(self.features):
            raise ValueError(
                "Expected {} model features, got {}".format(
                    len(self.features), len(feature_values)
                )
            )

        values = []
        for index, value in enumerate(feature_values):
            scale = self.scale[index]
            if scale == 0:
                scale = 1.0
            values.append((float(value) - self.mean[index]) / scale)

        for layer_index, layer in enumerate(self.layers):
            kernel = layer["kernel"]
            bias = layer["bias"]
            activation = layer["activation"]
            output = []
            for output_index, bias_value in enumerate(bias):
                total = float(bias_value)
                for input_index, input_value in enumerate(values):
                    total += input_value * float(kernel[input_index][output_index])
                if activation == "relu":
                    total = self._relu(total)
                elif activation == "sigmoid":
                    total = 1.0 / (1.0 + pow(2.718281828459045, -total))
                output.append(total)
            values = output

        prediction = max(0.0, min(1.0, values[0]))
        section_id = None
        for index, name in enumerate(self.features):
            if name == "section_id":
                section_id = str(int(feature_values[index]))
                break
        if section_id in self.section_caps:
            prediction = min(prediction, float(self.section_caps[section_id]))
        return prediction

