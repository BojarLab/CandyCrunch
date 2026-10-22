from pathlib import Path

import torch
from candycrunch.model import CandyCrunch_CNN, CandyCrunch_CNN2


def load_checkpoint(model_path, device=None, temperature=1.15):
    """Return keyword arguments for prediction.wrap_inference or wrap_inference_batch.

    model_path accepts a string or Path pointing to a training checkpoint.
    device defaults to CUDA when available, otherwise CPU. temperature is the
    inference calibration factor, independent of contrastive training temperature.
    The returned dictionary contains model, glycans, bin_num, comp_vector_order,
    and temperature. Pass it with wrap_inference(spectra_path, "N", **loaded).
    """
    checkpoint = torch.load(Path(model_path).expanduser(), map_location="cpu", weights_only=True)
    required = {"model_class", "model_kwargs", "state_dict", "glycans", "comp_vector_order"}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("Expected a training checkpoint containing " + ", ".join(sorted(required)))

    model_classes = {
        "CandyCrunch_CNN": CandyCrunch_CNN,
        "CandyCrunch_CNN2": CandyCrunch_CNN2,
    }
    model_class = checkpoint["model_class"]
    if model_class not in model_classes:
        raise ValueError(f"Unsupported model class: {model_class!r}")

    glycans = list(checkpoint["glycans"])
    comp_vector_order = list(checkpoint["comp_vector_order"])
    model_kwargs = checkpoint["model_kwargs"]
    if model_kwargs["num_classes"] != len(glycans):
        raise ValueError("Checkpoint vocabulary does not match num_classes")
    if model_kwargs["input_precursor_dim"] != len(comp_vector_order):
        raise ValueError("Checkpoint composition order does not match input_precursor_dim")

    model = model_classes[model_class](**model_kwargs)
    model.load_state_dict({name.removeprefix("module."): value for name, value in checkpoint["state_dict"].items()})
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return {
        "model": model,
        "glycans": glycans,
        "bin_num": model_kwargs["input_dim"],
        "comp_vector_order": comp_vector_order,
        "temperature": float(temperature),
    }
