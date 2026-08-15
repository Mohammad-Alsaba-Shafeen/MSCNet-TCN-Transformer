import mne
import numpy as np
import scipy.io as sio
from scipy.io import savemat
import os
import warnings
warnings.filterwarnings("ignore")

# ================= EA =================
def euclidean_alignment(data):
    x = data.astype(np.float64)
    n_trials, n_ch, n_t = x.shape

    cov_sum = np.zeros((n_ch, n_ch))
    for trial in x:
        cov_sum += trial @ trial.T / n_t
    R = cov_sum / n_trials

    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.maximum(eigvals, 1e-10)
    R_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    aligned = np.stack([R_inv_sqrt @ trial for trial in x])
    return aligned.astype(np.float32)

# ================= PATH =================
out_dir = './mymat_aligned/'
os.makedirs(out_dir, exist_ok=True)

# ================= TRAIN =================
print("Aligning Training Data...")
for nSub in range(1, 10):
    data_list = []
    label_list = []

    for nSes in range(1, 4):
        raw = mne.io.read_raw_gdf('./B0%d0%dT.gdf' % (nSub, nSes), preload=True, verbose=False)

        # Notch Filter
        raw.notch_filter(50., fir_design='firwin', verbose=False)

        events, event_dict = mne.events_from_annotations(raw, verbose=False)
        event_id = {
            'Left': event_dict['769'],
            'Right': event_dict['770']
        }
        selected_events = events[np.isin(events[:, 2], list(event_id.values()))]

        # REMOVE EOG, KEEP ONLY EEG (3 CHANNELS)
        raw.info['bads'] += ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']
        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')

        epochs = mne.Epochs(
            raw, selected_events, event_id,
            picks=picks,
            tmin=0, tmax=3.996,
            preload=True,
            baseline=None,
            verbose=False
        )

        data = epochs.get_data()[:, :, :1000]

        print(f"Sub {nSub} Ses {nSes} TRAIN shape:", data.shape)

        labels = sio.loadmat('./B0%d0%dT.mat' % (nSub, nSes))['classlabel']
        data_list.append(data)
        label_list.append(labels)

    # SAFE CONCAT
    data_sub = np.concatenate(data_list, axis=0)
    labels_sub = np.concatenate(label_list, axis=0)

    # Z-SCORE NORMALIZATION
    mean = data_sub.mean(axis=2, keepdims=True)
    std = data_sub.std(axis=2, keepdims=True)
    data_sub = ((data_sub - mean) / (std + 1e-6)).astype(np.float32)

    # EUCLIDEAN ALIGNMENT
    data_sub = euclidean_alignment(data_sub)

    print('B0%dT FINAL:' % nSub, data_sub.shape)
    savemat(out_dir + 'B0%dT.mat' % nSub, {'data': data_sub, 'label': labels_sub})

# ================= EVAL =================
print("\nAligning Evaluation Data...")
for nSub in range(1, 10):
    data_list = []
    label_list = []

    for nSes in range(4, 6):
        raw = mne.io.read_raw_gdf('./B0%d0%dE.gdf' % (nSub, nSes), preload=True, verbose=False)

        # Notch Filter
        raw.notch_filter(50., fir_design='firwin', verbose=False)

        events, event_dict = mne.events_from_annotations(raw, verbose=False)
        event_id = {'Unknown': event_dict['783']}
        selected_events = events[np.isin(events[:, 2], list(event_id.values()))]

        # REMOVE EOG, KEEP ONLY EEG (3 CHANNELS)
        raw.info['bads'] += ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']
        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')

        epochs = mne.Epochs(
            raw, selected_events, event_id,
            picks=picks,
            tmin=0, tmax=3.996,
            preload=True,
            baseline=None,
            on_missing='ignore',
            verbose=False
        )

        data = epochs.get_data()[:, :, :1000]

        print(f"Sub {nSub} Ses {nSes} EVAL shape:", data.shape)

        labels = sio.loadmat('./B0%d0%dE.mat' % (nSub, nSes))['classlabel']
        data_list.append(data)
        label_list.append(labels)

    # SAFE CONCAT
    data_sub = np.concatenate(data_list, axis=0)
    labels_sub = np.concatenate(label_list, axis=0)

    # NORMALIZATION
    mean = data_sub.mean(axis=2, keepdims=True)
    std = data_sub.std(axis=2, keepdims=True)
    data_sub = ((data_sub - mean) / (std + 1e-6)).astype(np.float32)

    # EA
    data_sub = euclidean_alignment(data_sub)

    print('B0%dE FINAL:' % nSub, data_sub.shape)
    savemat(out_dir + 'B0%dE.mat' % nSub, {'data': data_sub, 'label': labels_sub})
