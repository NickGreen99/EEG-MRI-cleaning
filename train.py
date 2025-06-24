import torch, math, itertools
from torch import optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader
import torch.multiprocessing as mp          #  ← already imported torch above

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from dataset_creation import EEGWindowStream
from res_unet import DeepDSP_UNetRes   # your model

def main():
        # ------------------------------------------------------------------
    #  Hyper-parameters
    # ------------------------------------------------------------------
    BATCH_SIZE        = 32
    WIN               = 512
    LR                = 1e-3
    EPOCHS            = 1
    BATCHES_PER_EPOCH = 1_000
    VAL_BATCHES       = 250

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AMP    = DEVICE.type == "cuda"

    # ------------------------------------------------------------------
    #  Datasets & DataLoaders
    # ------------------------------------------------------------------
    train_ds = EEGWindowStream("train",
                               win=WIN,
                               examples_per_epoch=BATCH_SIZE * BATCHES_PER_EPOCH)
    val_ds   = EEGWindowStream("val",
                               win=WIN,
                               examples_per_epoch=BATCH_SIZE * VAL_BATCHES)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        num_workers=4,              # keep >0 for speed
        prefetch_factor=4,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # ------------------------------------------------------------------
    #  Model, loss, optimiser
    # ------------------------------------------------------------------
    model   = DeepDSP_UNetRes(in_channels=2, out_channels=1).to(DEVICE)
    loss_fn = MSELoss()
    opt     = optim.Adam(model.parameters(), lr=LR)
    scaler  = torch.amp.GradScaler(device="cuda", enabled=AMP)  # ✓ new API

    # ------------------------------------------------------------------
    #  Logging helpers
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer    = SummaryWriter(f"runs/eeg_cleaning_{timestamp}")

    def run_epoch(epoch: int):
        model.train()
        running = 0.0

        train_iter = itertools.islice(train_loader, BATCHES_PER_EPOCH)
        for step, (x, y) in enumerate(train_iter, start=1):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda',enabled=AMP):
                y_hat = model(x)
                loss  = loss_fn(y_hat, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item()

            if step % 50 == 0:
                avg = running / 50
                print(f"[Epoch {epoch:03d} | step {step:04d}] loss = {avg:.5f}")
                global_step = epoch * BATCHES_PER_EPOCH + step
                writer.add_scalar("Loss/train", avg, global_step)
                running = 0.0

    def validate(epoch: int) -> float:
        model.eval()
        total = 0.0
        with torch.no_grad():
            val_iter = itertools.islice(val_loader, VAL_BATCHES)
            for x, y in val_iter:
                x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=AMP):
                    y_hat = model(x)
                    total += loss_fn(y_hat, y).item()

        vloss = total / VAL_BATCHES
        writer.add_scalar("Loss/val", vloss, epoch)
        print(f"→ validation loss: {vloss:.5f}")
        return vloss

    # ------------------------------------------------------------------
    #  Training loop
    # ------------------------------------------------------------------
    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
        run_epoch(epoch)
        vloss = validate(epoch)
        if vloss < best_val:
            best_val = vloss
            ckpt = f"model_{timestamp}_epoch{epoch}.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"✔ Saved new best model to {ckpt}")

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)   # explicit is safer
    main()
