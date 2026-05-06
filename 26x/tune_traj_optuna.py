# tune_traj_optuna.py
import os, math, random, numpy as np, torch, optuna
import torch.nn as nn
from torch.utils.data import DataLoader
from train_traj_seq2seq import TrajDataset, Seq2Seq, ade_fde, H_PAST, H_FUT, INPUT_SIZE, OUTPUT_SIZE  # 既存を利用

TRAIN_CSV = r"C:\Users\s1280\Desktop\trajectory_data\traj_dataset_p20_f45_s5_norm360x240_with_speed_train.csv"
VAL_CSV   = r"C:\Users\s1280\Desktop\trajectory_data\traj_dataset_p20_f45_s5_norm360x240_with_speed_val.csv"
SAVE_DIR  = "./optuna_ckpts"

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def tf_ratio_linear(epoch, total_epochs, start, end):
    a = max(0.0, min(1.0, 1 - epoch/(total_epochs-1 + 1e-9)))
    return end + (start - end) * a

@torch.no_grad()
def eval_epoch(model, loader, device, beta=0.02):
    model.eval()
    tot_loss = tot_ade = tot_fde = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x, tgt=None, tf_ratio=0.0)
        loss = nn.functional.smooth_l1_loss(pred, y, beta=beta)
        ade, fde = ade_fde(pred, y)
        bs = x.size(0); n += bs
        tot_loss += loss.item()*bs; tot_ade += ade.item()*bs; tot_fde += fde.item()*bs
    return tot_loss/n, tot_ade/n, tot_fde/n

def objective(trial: optuna.Trial):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== Search Space =====
    hidden  = trial.suggest_categorical("hidden", [96, 128, 160, 192, 256])
    layers  = trial.suggest_int("layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    lr      = trial.suggest_float("lr", 5e-4, 3e-3, log=True)
    wd      = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch   = trial.suggest_categorical("batch_size", [128, 192, 256, 384, 512])
    epochs  = trial.suggest_categorical("epochs", [15, 20, 25, 30])
    tf_start= trial.suggest_float("tf_start", 0.5, 0.95)
    tf_end  = trial.suggest_float("tf_end",   0.05, 0.5)
    beta    = trial.suggest_float("smoothl1_beta", 0.01, 0.1, log=True)

    # ===== Data =====
    train_ds = TrajDataset(TRAIN_CSV, H_PAST, H_FUT)
    val_ds   = TrajDataset(VAL_CSV,   H_PAST, H_FUT)
    train_ld = DataLoader(train_ds, batch_size=batch, shuffle=True,  num_workers=4, pin_memory=True, drop_last=True)
    val_ld   = DataLoader(val_ds,   batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)

    # ===== Model =====
    model = Seq2Seq()
    # 再構築してハイパラ反映
    from train_traj_seq2seq import Encoder, Decoder
    model.encoder = Encoder(input_size=INPUT_SIZE, hidden=hidden, n_layers=layers, dropout=dropout)
    model.decoder = Decoder(hidden=hidden, out_size=OUTPUT_SIZE, n_layers=layers, dropout=dropout)
    model.to(device)

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    best_val_ade = math.inf
    os.makedirs(SAVE_DIR, exist_ok=True)

    for ep in range(epochs):
        model.train()
        tf_ratio = tf_ratio_linear(ep, epochs, tf_start, tf_end)
        tot_loss = tot_ade = tot_fde = n = 0

        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True):
                pred = model(x, tgt=y, tf_ratio=tf_ratio)
                loss = nn.functional.smooth_l1_loss(pred, y, beta=beta)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()

            ade, fde = ade_fde(pred.detach(), y)
            bs = x.size(0); n += bs
            tot_loss += loss.item()*bs; tot_ade += ade.item()*bs; tot_fde += fde.item()*bs

        tr_loss = tot_loss/n; tr_ade = tot_ade/n; tr_fde = tot_fde/n
        va_loss, va_ade, va_fde = eval_epoch(model, val_ld, device, beta=beta)
        sched.step()

        # Optuna に中間報告
        trial.report(va_ade, step=ep)

        # プルーニング（改善が鈍ければ打ち切り）
        if trial.should_prune():
            raise optuna.TrialPruned()

        # ベスト保存
        if va_ade < best_val_ade:
            best_val_ade = va_ade
            torch.save({"state_dict": model.state_dict(),
                        "val_ade": va_ade, "val_fde": va_fde,
                        "trial_params": trial.params},
                       os.path.join(SAVE_DIR, f"best_trial_{trial.number}.pt"))

        print(f"[T{trial.number:02d} E{ep+1:02d}/{epochs}] "
              f"tr_loss {tr_loss:.4f} | tr_ADE {tr_ade:.4f} | tr_FDE {tr_fde:.4f} | "
              f"val_ADE {va_ade:.4f} | val_FDE {va_fde:.4f} | tf {tf_ratio:.2f} | lr {sched.get_last_lr()[0]:.2e}")

    return best_val_ade

if __name__ == "__main__":
    set_seed(42)
    # MedianPruner: 進捗の中央値より悪い試行を早期打ち切り
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5, n_min_trials=8)
    study = optuna.create_study(direction="minimize", pruner=pruner, study_name="traj_seq2seq")
    study.optimize(objective, n_trials=30, timeout=None, gc_after_trial=True)
    print("Best trial:", study.best_trial.number)
    print("Best val ADE:", study.best_value)
    print("Params:", study.best_trial.params)
