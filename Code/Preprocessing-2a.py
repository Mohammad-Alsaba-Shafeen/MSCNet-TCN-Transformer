import mne
import numpy as np
from scipy.io import savemat
import scipy.io as sio
import os


def euclidean_alignment(data):
    """
    Euclidean Alignment (EA) — from original TCANet paper preprocessing.

    Aligns each session's trials to a common reference covariance.
    Directly reduces session-to-session domain shift, which is the
    main reason val loss and test accuracy diverge for hard subjects
    (2, 4, 5, 6) in BCI IV-2a.

    Paper reference:
        He & Wu (2020) "Transfer Learning for Brain-Computer Interfaces:
        A Euclidean Space Data Alignment Approach"

    data: (trials, channels, time)  float32
    returns: aligned data same shape
    """
    x = data.astype(np.float64)   # (trials, 22, 1000)

    # ── Step 1: compute mean covariance across all trials ──────────
    n_trials, n_ch, n_t = x.shape
    cov_sum = np.zeros((n_ch, n_ch))
    for trial in x:
        cov_sum += trial @ trial.T / n_t
    R = cov_sum / n_trials          # (22, 22) mean covariance

    # ── Step 2: compute whitening matrix R^{-1/2} ──────────────────
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals          = np.maximum(eigvals, 1e-10)   # numerical stability
    R_inv_sqrt       = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    # ── Step 3: apply alignment to every trial ─────────────────────
    aligned = np.stack([R_inv_sqrt @ trial for trial in x])

    return aligned.astype(np.float32)


def changeGdf2Mat(dir_path, mode="train"):
    mode_str = 'T' if mode == "train" else 'E'

    for nSub in range(1, 10):
        print(f"Processing Subject A0{nSub}{mode_str}...")

        # ── Load file ──────────────────────────────────────────────
        data_filename = dir_path + f'A0{nSub}{mode_str}.gdf'
        raw = mne.io.read_raw_gdf(data_filename, preload=True)


        # ── Notch filter only — let MSCNet learn frequency features ─
        raw.notch_filter(50., fir_design='firwin')

        events, event_dict = mne.events_from_annotations(raw)

        if mode == "train":
            event_id = {
                'Left':   event_dict['769'],
                'Right':  event_dict['770'],
                'Foot':   event_dict['771'],
                'Tongue': event_dict['772']
            }
        else:
            event_id = {'Unknown': event_dict['783']}

        selected_events = events[np.isin(events[:, 2], list(event_id.values()))]

        # ── Index-based Channel Selection ──────────────────────────
        # Bypass channel names completely. BCI IV-2a has 25 channels.
        # The first 22 are EEG, the last 3 are EOG.
        picks = np.arange(22)

        # ── Epoch: 0–4s, exactly 1000 samples @ 250 Hz ────────────
        epochs = mne.Epochs(
            raw, selected_events, event_id,
            picks=picks,
            tmin=0.0, tmax=3.996,
            preload=True,
            baseline=None
        )

        data = epochs.get_data()[:, :, :1000]   # (trials, 22, 1000)
        assert data.shape[2] == 1000, \
            f"Expected 1000 samples, got {data.shape[2]}"

        # ── Step 1: Per-trial z-score per channel ──────────────────
        # Removes DC offset and amplitude differences across trials
        # without removing spectral content
        mean = data.mean(axis=2, keepdims=True)   # (trials, 22, 1)
        std  = data.std(axis=2, keepdims=True)    # (trials, 22, 1)
        data = ((data - mean) / (std + 1e-6)).astype(np.float32)

        # ── Step 2: Euclidean Alignment (EA) ──────────────────────
        # Applied per-session (train and eval separately) to reduce
        # session-to-session domain shift — the key missing piece
        # from the original TCANet paper preprocessing pipeline.
        # Biggest gains on hard subjects (2, 4, 5, 6).
        print(f"  Applying Euclidean Alignment...")
        final_data = euclidean_alignment(data)

        # ── Labels ─────────────────────────────────────────────────
        label_filename = dir_path + f'A0{nSub}{mode_str}.mat'
        mat            = sio.loadmat(label_filename)
        final_labels   = mat['classlabel']

        # ── Save ───────────────────────────────────────────────────
        out_path = dir_path + 'mymat_raw/'
        os.makedirs(out_path, exist_ok=True)
        result_filename = out_path + f'A0{nSub}{mode_str}.mat'
        savemat(result_filename, {'data': final_data, 'label': final_labels})
        print(f"  Saved {result_filename}  shape={final_data.shape}\n")


if __name__ == "__main__":
    dir_path = '/content/'
    changeGdf2Mat(dir_path, 'train')
    changeGdf2Mat(dir_path, 'eval')
