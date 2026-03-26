import numpy as np
import torch
from transformers import WhisperFeatureExtractor
from torch.utils.data import TensorDataset


# ================= NORMALIZATION =================
def normalize_lightcurves(curves):
    """
    Mean-center each light curve (preserve amplitude differences)
    """
    normalized = []
    for lc in curves:
        lc = lc.astype(np.float32)
        lc = lc - lc.mean()
        normalized.append(lc)
    return np.array(normalized, dtype=np.float32)


# ================= BINNING =================
def bin_to_length_mean(arr, target_len):
    """
    Shape-preserving binning via mean (for plotting / smoothing)
    """
    N = len(arr)
    edges = np.linspace(0, N, target_len + 1, dtype=int)

    binned = np.zeros(target_len, dtype=np.float64)
    for i in range(target_len):
        start, end = edges[i], edges[i+1]
        if end > start:
            binned[i] = np.mean(arr[start:end])
        else:
            binned[i] = arr[min(start, N-1)]

    return binned


# ================= WHISPER FEATURE BUILDING =================
def build_whisper_dataset(
    signal_curves,
    noise_curves,
    batch_size=100,
    sampling_rate=16000
):
    """
    Converts light curves into Whisper input features + labels
    """

    extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")

    # Normalize
    signal_norm = normalize_lightcurves(signal_curves)
    noise_norm = normalize_lightcurves(noise_curves)

    # Combine
    all_curves = list(noise_norm) + list(signal_norm)
    labels = [0]*len(noise_norm) + [1]*len(signal_norm)

    X_features_list = []

    for i in range(0, len(all_curves), batch_size):
        batch = all_curves[i:i+batch_size]

        inputs = extractor(
            batch,
            sampling_rate=sampling_rate,
            return_tensors="pt"
        )

        X_features_list.append(inputs.input_features)

    X_features = torch.cat(X_features_list, dim=0)
    y = torch.tensor(labels, dtype=torch.long)

    return TensorDataset(X_features, y)
