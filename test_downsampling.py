import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# # ------------------------------------------------------------
# # 1) load the mem-mapped file you just saved
# # ------------------------------------------------------------
# fname = "proc_memmap/016_654_1612_1000Hz.npy"   # adjust if needed
# mm    = np.load(fname, mmap_mode="r")           # shape (26, n_times)
# fs    = 1000                                   # Hz (your resample rate)

# # pick a representative channel, e.g. Cz = index 18
# chan  = 0
# sig   = mm[chan]                                # np.memmap slice, no RAM copy

# # ------------------------------------------------------------
# # 2) take a 10-second excerpt for the time-domain view
# # ------------------------------------------------------------
# t0_sec   = 60          # 1 min into the block
# dur      = 10          # seconds
# start    = int(t0_sec * fs)
# stop     = start + int(dur * fs)
# t        = np.arange(start, stop) / fs          # time axis in seconds
# excerpt  = sig[start:stop]

# # ------------------------------------------------------------
# # 3) compute single-sided PSD (Welch)
# # ------------------------------------------------------------
# def fft_psd(signal):
#   N       = len(signal)
#   freqs   = np.fft.rfftfreq(N, d=1/fs)        # 0 … +Nyquist
#   fft_val = np.fft.rfft(signal)
#   psd     = (np.abs(fft_val) ** 2) / N        # power spectrum
#   return freqs,psd

# f,Pxx = fft_psd(sig)

# # ------------------------------------------------------------
# # 4) plot
# # ------------------------------------------------------------
# plt.figure(figsize=(12, 6))

# # ---- time series -----------------------------------------------------------
# plt.subplot(2, 1, 1)
# plt.plot(t, excerpt, lw=0.7)
# plt.title(f"Cz — {dur}s excerpt (down-sampled to {fs} Hz)")
# plt.xlabel("Time [s]"); plt.ylabel("Amplitude [µV]")
# plt.grid(alpha=.3)

# # ---- PSD -------------------------------------------------------------------
# plt.subplot(2, 1, 2)
# plt.plot(f, Pxx)
# plt.title("Power spectral density (Welch)")
# plt.xlabel("Frequency [Hz]"); plt.ylabel("PSD")
# plt.xlim(0, fs/2)
# plt.ylim(0,0.0002)
# plt.grid(alpha=.3, which="both")

# plt.tight_layout(); plt.show()
import mne

tmin, tmax = 654.0, 1612.0 
CH_NAMES = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
            'F7','F8','T7','T8','P7','P8','FPz','Fz','Cz','Pz',
            'POz','Oz','FT9','FT10',"TP9'","TP10'"]
chan=0
fname = "split_data/train/016_mri.npy"
# --- load ORIGINAL 5 kHz slice (no resample) -----------------------------
raw_orig = mne.io.read_raw_brainvision(
              'data/016_scan.vhdr', preload=False, verbose="ERROR")
raw_orig.crop(tmin, tmax, include_tmax=False)
raw_orig.pick(CH_NAMES)
sig_orig = raw_orig.get_data(picks=chan)[0]          # 5 kHz trace

# --- load DOWNSAMPLED 1 kHz mem-map --------------------------------------
sig_ds   = np.load(fname, mmap_mode="r")[chan]       # 1 kHz trace

# --- use the SAME PSD estimator -----------------------------------------
def psd_fft(sig, fs):
    freqs = np.fft.rfftfreq(len(sig), 1/fs)
    psd   = np.abs(np.fft.rfft(sig))**2 / len(sig)
    return freqs, psd

f1, P1 = psd_fft(sig_orig, 5000)
f2, P2 = psd_fft(sig_ds,   1000)

# --- plot ---------------------------------------------------------------
plt.figure(figsize=(10,4))
plt.semilogy(f1, P1, label="5 kHz raw")
plt.semilogy(f2, P2, label="1 kHz resampled")
plt.xlim(0, 2500)
plt.xlabel("Frequency (Hz)"); plt.ylabel("PSD (log)")
plt.title("PSD before vs. after down-sampling")
plt.legend(); plt.grid(True, ls="--", alpha=.3)
plt.show()
