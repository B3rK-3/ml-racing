"""Load self-contained ONNX policies for ghost-car time trials."""

from pathlib import Path

import numpy as np


class OnnxPolicy:
    """Run a fixed float32 observation/action graph on the CPU only."""

    _MAX_MODEL_BYTES = 100 * 1024 * 1024

    def __init__(self, path):
        try:
            import onnx
            import onnxruntime as ort
            from google.protobuf.message import Message
        except ImportError as exc:
            raise ValueError(
                "ONNX policies require the 'onnx' and 'onnxruntime' packages"
            ) from exc

        path = Path(path)
        try:
            if path.stat().st_size > self._MAX_MODEL_BYTES:
                raise ValueError("ONNX model exceeds the 100 MiB size limit")
            with path.open("rb") as model_file:
                model_bytes = model_file.read(self._MAX_MODEL_BYTES + 1)
            if len(model_bytes) > self._MAX_MODEL_BYTES:
                raise ValueError("ONNX model exceeds the 100 MiB size limit")
            # Parsing bytes does not resolve or load external tensor files.
            model = onnx.load_model_from_string(model_bytes)

            def reject_external_data(message):
                if isinstance(message, onnx.TensorProto):
                    if message.external_data or message.data_location == onnx.TensorProto.EXTERNAL:
                        raise ValueError(
                            "ONNX external tensor data is not allowed; use one self-contained file"
                        )
                # Include nested graphs, tensor attributes, sparse initializers,
                # local functions, and training graphs, not just the root graph.
                for field, value in message.ListFields():
                    if field.message_type is not None:
                        if isinstance(value, Message):
                            reject_external_data(value)
                        else:
                            for item in value:
                                reject_external_data(item)

            reject_external_data(model)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            self._session = ort.InferenceSession(
                model_bytes,
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            inputs = self._session.get_inputs()
            outputs = self._session.get_outputs()
            if (
                len(inputs) != 1
                or inputs[0].name != "obs"
                or inputs[0].type != "tensor(float)"
                or inputs[0].shape != [1, 156]
                or self._session.get_overridable_initializers()
            ):
                raise ValueError(
                    "ONNX policy must have exactly one input: obs, tensor(float), shape [1, 156]"
                )
            if (
                len(outputs) != 1
                or outputs[0].name != "action"
                or outputs[0].type != "tensor(float)"
                or outputs[0].shape != [1, 2]
            ):
                raise ValueError(
                    "ONNX policy must have exactly one output: action, tensor(float), shape [1, 2]"
                )
        except Exception as exc:
            raise ValueError("Cannot load ONNX policy '{}': {}".format(path, exc)) from exc

    def act(self, obs):
        """Return normalized [steering, target speed] for four stacked frames."""
        if not isinstance(obs, np.ndarray) or obs.dtype != np.float32 or obs.shape != (1, 156):
            raise ValueError("Policy observation must be a float32 ndarray with shape (1, 156)")
        if not np.isfinite(obs).all():
            raise ValueError("Policy observation contains nonfinite values")
        frames = obs.reshape(4, 39)
        if (
            np.any(frames[:, :36] < 0)
            or np.any(frames[:, :36] > 1)
            or np.any(frames[:, 36:] < -1)
            or np.any(frames[:, 36:] > 1)
        ):
            raise ValueError("Policy observation is outside its normalized feature ranges")
        try:
            action = self._session.run(["action"], {"obs": obs})[0]
        except Exception as exc:
            raise ValueError("ONNX policy inference failed: {}".format(exc)) from exc
        if (
            not isinstance(action, np.ndarray)
            or action.dtype != np.float32
            or action.shape != (1, 2)
        ):
            raise ValueError("ONNX policy action must be float32 with shape (1, 2)")
        if not np.isfinite(action).all():
            raise ValueError("ONNX policy action contains nonfinite values")
        action = action[0]
        action[0] = np.clip(action[0], -1.0, 1.0)
        action[1] = np.clip(action[1], 0.0, 1.0)
        return action
