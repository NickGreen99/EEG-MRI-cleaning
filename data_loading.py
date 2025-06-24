from pathlib import Path
import random, gc
import mne, numpy as np

# ---------------------------------------------------------------------
#  CONFIG
# ---------------------------------------------------------------------
SUBJECTS = ["016", "017", "019", "022", "023"]      # ← your IDs

RAW_DIR   = Path("data_raw")        # *.vhdr live here
OUT_DIR   = Path("data")            # one sub-folder per subject
OUT_DIR.mkdir(exist_ok=True)

# EEG channels you care about (order matters!)
CH_NAMES = [
    'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
    'F7','F8','T7','T8','P7','P8','FPz','Fz','Cz','Pz',
    'POz','Oz','FT9','FT10',"TP9'","TP10'"
]

# Time windows (seconds) inside each scan
TMIN_MRI   = 654.0      # start of contaminated segment
TMAX_MRI   = 1612.0     # end   of contaminated segment
CLEAN_SEC  = 120.0      # 2-min clean block immediately before TMIN_MRI
BLOCKS = {
    "eeg_clean": (TMIN_MRI - CLEAN_SEC, TMIN_MRI),   # clean EEG
    "eeg_dirty": (TMIN_MRI, TMAX_MRI),               # artefact-only*
}
FS_OUT = 1_000          # resample to 1000 Hz
# ---------------------------------------------------------------------
#  *If your “dirty” window still contains EEG, treat it as artefact-heavy
#   or apply a high-pass to isolate EMI before saving.
# ---------------------------------------------------------------------

def export_block(vhdr_path: Path, sid: str,
                 t0: float, t1: float,
                 block_name: str, subj_dir: Path):
    """Crop → pick channels → resample → save as <block_name>.npy"""
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose="ERROR")
    raw.crop(t0, t1, include_tmax=False)
    raw.pick(CH_NAMES)
    raw.load_data(verbose="ERROR")
    raw.resample(FS_OUT, npad="auto", verbose="ERROR")

    subj_dir.mkdir(parents=True, exist_ok=True)
    out_path = subj_dir / f"{block_name}.npy"
    np.save(out_path, raw.get_data().astype(np.float32))

    print(f"[{sid}] {block_name:9} → {out_path.relative_to(Path.cwd())}")

    raw.close()
    del raw
    gc.collect()

def process_subject(sid: str):
    vhdr = RAW_DIR / f"{sid}_scan.vhdr"
    if not vhdr.exists():
        print(f"[SKIP] {vhdr} not found")
        return
    subj_dir = OUT_DIR / sid
    for block_name, (t0, t1) in BLOCKS.items():
        export_block(vhdr, sid, t0, t1, block_name, subj_dir)

# ---------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------
def main():
    print("Creating per-subject folders in", OUT_DIR.resolve())
    for sid in SUBJECTS:
        process_subject(sid)

if __name__ == "__main__":
    main()
