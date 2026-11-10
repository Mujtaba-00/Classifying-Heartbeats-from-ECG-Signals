import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt

# Project paths
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# These names match the Assignment 3 notebook.
CLASS_NAMES = ["N", "S", "V", "F", "Q"]
N_FEATURES = 187
DEFAULT_TEST_CSV = BASE_DIR / "mitbih_test.csv"
DEFAULT_CHECKPOINT = BASE_DIR / "results" / "best_model_0.01.pth"


st.set_page_config(
    page_title="ECG Heartbeat Classifier",
    page_icon="",
    layout="wide",
)

st.title("ECG Heartbeat Classification")
st.write(
    "Select a heartbeat from the test set or upload a CSV containing one "
    "heartbeat represented by 187 values. The trained ECGNet model will "
    "predict one of five heartbeat classes."
)


@st.cache_resource
def load_model(checkpoint_path: str, model_file_stamp: float):
    """Load the ECGNet architecture and trained weights."""
    try:
        from model import ECGNet
    except Exception as exc:
        raise RuntimeError(
            "Could not import ECGNet. Put your original model.py file "
            "inside the app's src/ folder."
        ) from exc

    net = ECGNet()
    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older PyTorch versions
        state = torch.load(checkpoint_path, map_location="cpu")

    # Support either a raw state_dict or a checkpoint containing one.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    net.load_state_dict(state)
    net.eval()
    return net


def read_test_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, header=None)
    else:
        if not DEFAULT_TEST_CSV.exists():
            return None
        data = pd.read_csv(DEFAULT_TEST_CSV, header=None)

    if data.shape[1] < N_FEATURES:
        raise ValueError(
            f"Expected at least {N_FEATURES} columns, but found {data.shape[1]}."
        )

    # The notebook's test file has 187 signal columns and a label in column 188.
    return data


def parse_single_heartbeat(uploaded_file):
    raw = pd.read_csv(uploaded_file, header=None)
    if raw.shape[0] != 1 or raw.shape[1] != N_FEATURES:
        raise ValueError(
            "The uploaded heartbeat CSV must contain exactly one row "
            f"and {N_FEATURES} columns (no class-label column). "
            f"Your file has {raw.shape[0]} row(s) and {raw.shape[1]} column(s)."
        )
    values = raw.iloc[0].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("The CSV contains missing or non-numeric values.")
    return values


def predict(net, signal):
    x = torch.tensor(signal, dtype=torch.float32).reshape(1, N_FEATURES)
    with torch.no_grad():
        logits = net(x)