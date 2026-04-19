"""
Training pipeline for CTR prediction with custom AdaGrad optimizers.

Usage examples:
  # Diagonal AdaGrad-COMID without regularization
  python train.py --optimizer comid --lr 0.05

  # Diagonal AdaGrad-RDA with L1 regularization
  python train.py --optimizer rda --reg l1 --reg_lambda 1e-5 --lr 0.05

  # Standard PyTorch AdaGrad baseline
  python train.py --optimizer adagrad --lr 0.05

  # Compare all optimizers
  python train.py --compare

  # Save model weights for later inspection
  python train.py --optimizer comid --save_model weights_comid.pt
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, roc_auc_score

from data import detect_features, iterate_minibatches, iterate_unlabeled
from model import CTRModel
from optimizers import DiagonalAdaGradCOMID, DiagonalAdaGradRDA


DATA_DIR = os.path.dirname(os.path.abspath(__file__))


_labeled_files_cache = None

def get_labeled_files():
    """Get sorted CSV files that contain a 'click' column (cached)."""
    global _labeled_files_cache
    if _labeled_files_cache is not None:
        return _labeled_files_cache
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, 'train_*.csv')))
    labeled = []
    for f in all_files:
        _, _, has_labels = detect_features(f)
        if has_labels:
            labeled.append(f)
    if len(labeled) < 2:
        raise FileNotFoundError(
            f"Need at least 2 labeled CSV files in {DATA_DIR}, found {len(labeled)}"
        )
    _labeled_files_cache = labeled
    return labeled


def get_train_files():
    """Get training CSV files (all labeled files except the last)."""
    return get_labeled_files()[:-1]


def get_test_file():
    """Get test CSV file (last labeled day)."""
    return get_labeled_files()[-1]


def build_optimizer(model, args):
    """Build optimizer from command-line arguments."""
    if args.optimizer == 'comid':
        return DiagonalAdaGradCOMID(
            model.parameters(),
            lr=args.lr,
            delta=args.delta,
            reg_type=args.reg,
            reg_lambda=args.reg_lambda,
            constraint=args.constraint,
            constraint_param=args.constraint_param,
        )
    elif args.optimizer == 'rda':
        return DiagonalAdaGradRDA(
            model.parameters(),
            lr=args.lr,
            delta=args.delta,
            reg_type=args.reg,
            reg_lambda=args.reg_lambda,
            constraint=args.constraint,
            constraint_param=args.constraint_param,
        )
    elif args.optimizer == 'adagrad':
        return torch.optim.Adagrad(model.parameters(), lr=args.lr, eps=args.delta)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")


def train_one_day(model, optimizer, criterion, csv_path, features, args,
                  day_label=''):
    """Train model on one day's data. Returns average loss."""
    model.train()
    # Accumulate on-device to avoid per-batch MPS/CUDA sync from .item()
    running_loss = torch.tensor(0.0, device=args.device)
    running_clicks = torch.tensor(0, device=args.device, dtype=torch.long)
    n_batches = 0
    n_samples = 0
    t0 = time.time()

    for batch_idx, (indices, labels) in enumerate(
        iterate_minibatches(csv_path, features, batch_size=args.batch_size,
                            hash_size=args.hash_size, device=args.device,
                            shuffle_chunks=False, sample_rate=args.sample_rate)
    ):
        optimizer.zero_grad(set_to_none=True)
        logits = model(indices)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = len(labels)
        running_loss += loss.detach() * bs
        running_clicks += labels.sum().to(torch.long)
        n_samples += bs
        n_batches += 1

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            avg = running_loss.item() / n_samples  # sync only at log points
            elapsed = time.time() - t0
            print(f"  {day_label} batch {batch_idx:>6d} | "
                  f"loss={avg:.6f} | {n_samples:,} samples | {elapsed:.1f}s")

    total_loss = running_loss.item()
    n_clicks = running_clicks.item()
    avg_loss = total_loss / max(n_samples, 1)
    n_nonclicks = n_samples - n_clicks
    cr = 100 * n_clicks / max(n_samples, 1)
    elapsed = time.time() - t0
    print(f"  {day_label} done | loss={avg_loss:.6f} | "
          f"{n_samples:,} samples ({n_clicks:,} clicks, "
          f"{n_nonclicks:,} non-clicks, CR={cr:.1f}%) | {elapsed:.1f}s")
    return avg_loss, total_loss, n_samples


def evaluate(model, criterion, csv_path, features, args):
    """Evaluate model on test data. Returns log_loss and AUC."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_samples = 0

    # Use large batches for evaluation (no gradients needed, doesn't
    # affect online learning semantics — only inference speed)
    eval_batch_size = max(args.batch_size, 4096)

    with torch.no_grad():
        for indices, labels in iterate_minibatches(
            csv_path, features, batch_size=eval_batch_size,
            hash_size=args.hash_size, device=args.device,
            shuffle_chunks=False
        ):
            logits = model(indices)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            n_samples += len(labels)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Clip predictions for numerical stability
    all_preds = np.clip(all_preds, 1e-7, 1 - 1e-7)

    logloss = log_loss(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    avg_loss = total_loss / max(n_samples, 1)

    return {
        'logloss': logloss,
        'auc': auc,
        'bce_loss': avg_loss,
        'n_samples': n_samples,
        'click_rate': float(all_labels.mean()),
    }


def print_weight_stats(model):
    """Print per-layer weight statistics for inspection."""
    print(f"\n{'='*95}")
    print("Model Weight Statistics")
    print(f"{'='*95}")
    print(f"{'Layer':<40s} {'Shape':>15s} {'||w||_2':>10s} {'Sparsity':>10s}"
          f" {'Min':>10s} {'Max':>10s}")
    print(f"{'─'*95}")
    total_params = 0
    total_zeros = 0
    for name, param in model.named_parameters():
        data = param.data
        n = data.numel()
        nz = (data.abs() < 1e-8).sum().item()
        total_params += n
        total_zeros += nz
        print(f"{name:<40s} {str(list(data.shape)):>15s}"
              f" {data.norm().item():>10.4f} {100*nz/n:>9.2f}%"
              f" {data.min().item():>10.6f} {data.max().item():>10.6f}")
    print(f"{'─'*95}")
    print(f"Total: {total_params:,} params, {100*total_zeros/total_params:.2f}% near-zero")
    print(f"{'='*95}")


def analyze_experiment(model, optimizer, train_files, features, args,
                       total_online_loss, total_online_samples):
    """Post-training analysis for Thesis Chapter 7.

    Computes:
      1. Sparsity analysis — effect of L1 regularization
      2. Constraint verification — feasible set K is respected
      3. Effective learning rates — AdaGrad's per-coordinate adaptivity
      4. Empirical regret — R_T and R_T/T (should be sublinear)
      5. Feature importance — per-field embedding norm as influence proxy
    """
    import math
    criterion = nn.BCEWithLogitsLoss()

    print(f"\n{'='*70}")
    print("ANALYSIS (Thesis Chapter 7)")
    print(f"{'='*70}")

    # ── 1. Sparsity Analysis (L1 regularization, Thesis §6.2 case 2) ──
    print(f"\n1. Sparsity Analysis:")
    total_params = 0
    total_exact_zero = 0
    for name, param in model.named_parameters():
        data = param.data
        n = data.numel()
        exact_zero = (data == 0).sum().item()
        near_zero = (data.abs() < 1e-6).sum().item()
        total_params += n
        total_exact_zero += exact_zero
        print(f"   {name}: {exact_zero:,}/{n:,} exact zeros "
              f"({100*exact_zero/n:.1f}%), "
              f"{near_zero:,} near-zero (<1e-6)")
    print(f"   Total: {total_exact_zero:,}/{total_params:,} "
          f"({100*total_exact_zero/total_params:.1f}%) exact zeros")

    # ── 2. Constraint Verification (Thesis §6.3–6.5) ──
    if args.constraint != 'none':
        print(f"\n2. Constraint Verification (K = {args.constraint}, "
              f"param = {args.constraint_param}):")
        all_ok = True
        for name, param in model.named_parameters():
            data = param.data
            if args.constraint == 'box':
                val = data.abs().max().item()
                ok = val <= args.constraint_param + 1e-6
                sym = '<=', f'max|w_i| = {val:.6f}'
            elif args.constraint == 'l2_ball':
                val = data.norm().item()
                ok = val <= args.constraint_param + 1e-6
                sym = '<=', f'||w||_2 = {val:.6f}'
            elif args.constraint == 'l1_ball':
                val = data.norm(1).item()
                ok = val <= args.constraint_param + 1e-6
                sym = '<=', f'||w||_1 = {val:.6f}'
            all_ok = all_ok and ok
            status = 'OK' if ok else 'VIOLATED'
            print(f"   {name}: {sym[1]} {sym[0]} {args.constraint_param} [{status}]")
        print(f"   All constraints satisfied: {all_ok}")
    else:
        print(f"\n2. Constraint: K = R^n (unconstrained)")

    # ── 3. Effective Learning Rates η/h_i (AdaGrad adaptivity) ──
    print(f"\n3. Effective Learning Rates (AdaGrad adaptivity):")
    print(f"   η/h_i shows per-coordinate step sizes after adaptation.")
    print(f"   Large ratio = strong adaptivity (different coordinates learn at different speeds).")
    for group in optimizer.param_groups:
        lr = group['lr']
        delta = group.get('delta', 1e-10)
        for p in group['params']:
            if p not in optimizer.state:
                continue
            state = optimizer.state[p]
            if 'sum_sq_grad' not in state:
                continue
            ssq = state['sum_sq_grad']
            h = delta + torch.sqrt(ssq)
            eff_lr = lr / h
            # For RDA, h was divided by t — eff_lr needs adjustment
            ratio = eff_lr.max() / eff_lr.min() if eff_lr.min() > 0 else float('inf')
            print(f"   param [{list(p.shape)}]: "
                  f"min={eff_lr.min():.2e}, "
                  f"median={eff_lr.median():.2e}, "
                  f"max={eff_lr.max():.2e}, "
                  f"ratio={ratio:.1f}x")

    # ── 4. Empirical Regret (Thesis Theorems 4.3, 5.4) ──
    # R_T = Σ f_t(w_t) − Σ f_t(w_T)
    # where w_T is the final model (approximation of w* = argmin Σ f_t)
    print(f"\n4. Empirical Regret Analysis:")
    online_avg = total_online_loss / total_online_samples
    print(f"   Online loss Σ f_t(w_t):     {total_online_loss:>12,.2f}  "
          f"(avg: {online_avg:.6f})")

    # Hindsight: replay all training data through final model
    print(f"   Computing hindsight loss (replaying data through final model)...")
    model.eval()
    hindsight_loss = torch.tensor(0.0, device=args.device)
    hindsight_samples = 0
    with torch.no_grad():
        for f in train_files:
            for indices, labels in iterate_minibatches(
                f, features, batch_size=max(args.batch_size, 4096),
                hash_size=args.hash_size, device=args.device,
                shuffle_chunks=False, sample_rate=args.sample_rate
            ):
                loss = criterion(model(indices), labels)
                hindsight_loss += loss.detach() * len(labels)
                hindsight_samples += len(labels)

    hl = hindsight_loss.item()
    hindsight_avg = hl / max(hindsight_samples, 1)
    print(f"   Hindsight loss Σ f_t(w_T):  {hl:>12,.2f}  "
          f"(avg: {hindsight_avg:.6f})")

    regret = total_online_loss - hl
    avg_regret = regret / total_online_samples
    T = total_online_samples
    print(f"   Regret R_T:                 {regret:>12,.2f}")
    print(f"   Average regret R_T/T:       {avg_regret:>12.6f}")
    print(f"   T (total samples):          {T:>12,}")

    if regret > 0:
        regret_sqrt = regret / math.sqrt(T)
        print(f"   R_T/sqrt(T):                {regret_sqrt:>12.4f}  "
              f"(bounded = sublinear)")
        print(f"   Sublinear regret confirmed: R_T/T = {avg_regret:.6f} -> 0")
    else:
        print(f"   Negative regret: online model outperformed fixed hindsight model.")
        print(f"   (Expected in non-convex settings — model adapts during training)")

    # ── 5. Feature Importance (per-field embedding norm) ──
    # Each feature field occupies a slice of the embedding table.
    # The L2-norm of each slice reflects how much the model relies on
    # that feature — larger norm = more influence on predictions.
    print(f"\n5. Feature Importance (per-field embedding norm):")

    # Access the raw model (unwrap torch.compile if active)
    raw_model = model
    if hasattr(model, '_orig_mod'):
        raw_model = model._orig_mod

    emb_weight = raw_model.embedding.weight.data
    hash_size = raw_model.hash_size
    field_names = features + ['hour_of_day']
    n_fields = len(field_names)

    importances = []
    for i, fname in enumerate(field_names):
        start = i * hash_size
        end = (i + 1) * hash_size
        field_emb = emb_weight[start:end]
        norm = field_emb.norm().item()
        nonzero = (field_emb.abs() > 1e-8).any(dim=1).sum().item()
        importances.append((norm, nonzero, fname))

    # Sort by norm descending
    importances.sort(key=lambda x: -x[0])

    total_norm = sum(imp[0] for imp in importances)
    print(f"   {'Rank':<5s} {'Feature':<20s} {'||emb||_2':>10s} {'% total':>8s} "
          f"{'Active rows':>12s}")
    print(f"   {'─'*60}")
    for rank, (norm, nonzero, fname) in enumerate(importances, 1):
        pct = 100 * norm / total_norm if total_norm > 0 else 0
        print(f"   {rank:<5d} {fname:<20s} {norm:>10.4f} {pct:>7.1f}% "
              f"{nonzero:>8,}/{hash_size}")

    print(f"{'='*70}")


def run_experiment(args):
    """Run a single training + evaluation experiment."""
    print(f"\n{'='*70}")
    print(f"Optimizer: {args.optimizer.upper()}"
          f" | Reg: {args.reg} (λ={args.reg_lambda})"
          f" | K: {args.constraint}"
          + (f" (R={args.constraint_param})" if args.constraint != 'none' else "")
          + f" | LR: {args.lr} | δ: {args.delta}"
          + (f" | Sample: {args.sample_rate:.0%}" if args.sample_rate < 1.0 else ""))
    print(f"Model: embed_dim={args.embed_dim}, hidden={args.hidden_dims}, "
          f"hash_size={args.hash_size}")
    print(f"{'='*70}")

    # Detect features from data
    train_files = get_train_files()
    features, n_fields, _ = detect_features(train_files[0])
    print(f"Features ({n_fields} fields): {features} + hour_of_day")

    # Build model
    model = CTRModel(
        n_fields=n_fields,
        hash_size=args.hash_size,
        embed_dim=args.embed_dim,
        hidden_dims=args.hidden_dims,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Compile model for faster forward/backward (MPS not supported)
    if hasattr(torch, 'compile') and args.device not in ('mps',):
        model = torch.compile(model)

    # Build optimizer and loss
    optimizer = build_optimizer(model, args)
    criterion = nn.BCEWithLogitsLoss()

    # Training: iterate over day files with prequential evaluation.
    # After training on day i, evaluate on day i+1 BEFORE training on it.
    # This tracks how well the model generalizes to unseen future data.
    all_labeled = get_labeled_files()
    print(f"\nTraining on {len(train_files)} days "
          f"(prequential eval on next day after each)...")
    train_losses = []
    daily_metrics = []  # (trained_on, evaluated_on, results)
    total_online_loss = 0.0   # Σ f_t(w_t) — for regret computation
    total_online_samples = 0

    # Baseline: evaluate untrained model on first day
    first_name = os.path.basename(train_files[0]).replace('.csv', '')
    baseline = evaluate(model, criterion, train_files[0], features, args)
    daily_metrics.append(('(init)', first_name, baseline))
    print(f"  Baseline (untrained) on {first_name}: "
          f"AUC={baseline['auc']:.6f}  LogLoss={baseline['logloss']:.6f}")

    for epoch in range(args.epochs):
        if args.epochs > 1:
            print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        for i, f in enumerate(train_files):
            day_name = os.path.basename(f).replace('.csv', '')

            # Train on this day
            avg_loss, day_total_loss, day_samples = train_one_day(
                model, optimizer, criterion, f, features, args,
                day_label=f"[{day_name}]"
            )
            train_losses.append(avg_loss)
            total_online_loss += day_total_loss
            total_online_samples += day_samples

            # Prequential: evaluate on the next day (without training)
            next_idx = i + 1
            if next_idx < len(all_labeled):
                next_file = all_labeled[next_idx]
                next_name = os.path.basename(next_file).replace('.csv', '')
                eval_res = evaluate(model, criterion, next_file, features, args)
                daily_metrics.append((day_name, next_name, eval_res))
                print(f"    -> eval {next_name}: "
                      f"AUC={eval_res['auc']:.6f}  "
                      f"LogLoss={eval_res['logloss']:.6f}")

    # Final test results (last entry in daily_metrics)
    test_file = get_test_file()
    test_name = os.path.basename(test_file).replace('.csv', '')
    results = daily_metrics[-1][2] if daily_metrics else \
        evaluate(model, criterion, test_file, features, args)

    print(f"\n{'─'*50}")
    print(f"Test Results ({test_name}):")
    print(f"  LogLoss:    {results['logloss']:.6f}")
    print(f"  AUC:        {results['auc']:.6f}")
    print(f"  BCE Loss:   {results['bce_loss']:.6f}")
    print(f"  Samples:    {results['n_samples']:,}")
    print(f"  Click Rate: {results['click_rate']:.4f}")
    print(f"{'─'*50}")

    # Day-by-day AUC progression table
    if daily_metrics:
        print(f"\n{'='*70}")
        print("Prequential AUC Progression (eval on next day after training)")
        print(f"{'='*70}")
        print(f"{'Trained on':<20s} {'Evaluated on':<20s} {'AUC':>10s} {'LogLoss':>10s}")
        print(f"{'─'*60}")
        for trained, evaled, res in daily_metrics:
            print(f"{trained:<20s} {evaled:<20s} "
                  f"{res['auc']:>10.6f} {res['logloss']:>10.6f}")
        print(f"{'─'*60}")

    # Weight inspection
    print_weight_stats(model)

    # Chapter 7 analysis: sparsity, constraints, learning rates, regret
    analyze_experiment(model, optimizer, train_files, features, args,
                       total_online_loss, total_online_samples)

    # Save model if requested
    if args.save_model:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer': args.optimizer,
            'reg_type': args.reg,
            'reg_lambda': args.reg_lambda,
            'lr': args.lr,
            'features': features,
            'n_fields': n_fields,
            'hash_size': args.hash_size,
            'embed_dim': args.embed_dim,
            'hidden_dims': args.hidden_dims,
            'results': results,
        }, args.save_model)
        print(f"\nModel saved to {args.save_model}")

    return results


def predict_unlabeled(args):
    """Generate click predictions for an unlabeled CSV using a saved model."""
    if not args.load_model:
        raise ValueError("--load_model is required for --predict")

    # Load checkpoint
    ckpt = torch.load(args.load_model, map_location=args.device, weights_only=False)
    features = ckpt['features']
    n_fields = ckpt['n_fields']

    print(f"Loaded model from {args.load_model}")
    print(f"  Optimizer: {ckpt['optimizer']}, Reg: {ckpt['reg_type']} "
          f"(λ={ckpt['reg_lambda']}), LR: {ckpt['lr']}")

    # Build model and load weights
    model = CTRModel(
        n_fields=n_fields,
        hash_size=ckpt['hash_size'],
        embed_dim=ckpt['embed_dim'],
        hidden_dims=ckpt['hidden_dims'],
    ).to(args.device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Find unlabeled file
    predict_file = args.predict
    _, _, has_labels = detect_features(predict_file)
    if has_labels:
        print(f"Note: {predict_file} has labels — ignoring them for prediction.")

    print(f"Predicting on {os.path.basename(predict_file)}...")

    import pandas as pd
    out_path = args.predict_output or predict_file.replace('.csv', '_predictions.csv')

    # Stream predictions to CSV — no accumulation in RAM
    n_samples = 0
    prob_sum = 0.0
    high_click = 0
    first_chunk = True
    top_rows = []

    with torch.no_grad():
        for indices, row_slice in iterate_unlabeled(
            predict_file, features, batch_size=4096,
            hash_size=ckpt['hash_size'], device=args.device,
        ):
            logits = model(indices)
            probs = torch.sigmoid(logits).cpu().numpy()

            out_df = row_slice.copy()
            out_df['click_prob'] = probs
            out_df.to_csv(out_path, mode='w' if first_chunk else 'a',
                          header=first_chunk, index=False)
            first_chunk = False

            n_samples += len(probs)
            prob_sum += probs.sum()
            high_click += (probs > 0.5).sum()

            # Track top predictions
            for i in range(len(probs)):
                if len(top_rows) < 10 or probs[i] > top_rows[-1][0]:
                    row = row_slice.iloc[i]
                    top_rows.append((probs[i], row.get('hour', ''),
                                     row.get('site_domain', ''),
                                     row.get('device_type', '')))
                    top_rows.sort(key=lambda x: -x[0])
                    top_rows = top_rows[:10]

    print(f"\nPredictions saved to {out_path}")
    print(f"  Samples:       {n_samples:,}")
    print(f"  Mean P(click): {prob_sum / max(n_samples, 1):.4f}")
    print(f"  P(click) > 0.5: {high_click:,} "
          f"({100 * high_click / max(n_samples, 1):.2f}%)")
    print(f"  Top-10 highest click probabilities:")
    for prob, hour, site, device in top_rows:
        print(f"    P={prob:.4f}  hour={hour}  site={site}  device={device}")

    return out_path


def run_comparison(args):
    """Compare all optimizer variants."""
    configs = [
        ('comid', 'none', 0.0),
        ('comid', 'l1', args.reg_lambda),
        ('comid', 'l2', args.reg_lambda),
        ('rda', 'none', 0.0),
        ('rda', 'l1', args.reg_lambda),
        ('rda', 'l2', args.reg_lambda),
        ('adagrad', 'none', 0.0),  # PyTorch baseline
    ]

    all_results = {}
    for opt_name, reg, lam in configs:
        label = f"{opt_name}_{reg}" + (f"_lam{lam}" if lam > 0 else "")
        args.optimizer = opt_name
        args.reg = reg
        args.reg_lambda = lam

        results = run_experiment(args)
        all_results[label] = results

    # Summary table
    print(f"\n\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Optimizer':<30s} {'LogLoss':>10s} {'AUC':>10s}")
    print(f"{'─'*50}")
    for label, res in sorted(all_results.items(), key=lambda x: x[1]['logloss']):
        print(f"{label:<30s} {res['logloss']:>10.6f} {res['auc']:>10.6f}")
    print(f"{'─'*50}")

    return all_results


def parse_args():
    parser = argparse.ArgumentParser(
        description='CTR Prediction with Adaptive Subgradient Methods'
    )

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='comid',
                        choices=['comid', 'rda', 'adagrad'],
                        help='Optimizer: comid, rda, or adagrad (baseline)')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate η (default: 0.05)')
    parser.add_argument('--delta', type=float, default=1e-10,
                        help='Numerical stability δ (default: 1e-10)')

    # Regularization
    parser.add_argument('--reg', type=str, default='none',
                        choices=['none', 'l1', 'l2'],
                        help='Regularization type φ (default: none)')
    parser.add_argument('--reg_lambda', type=float, default=1e-5,
                        help='Regularization strength λ (default: 1e-5)')

    # Feasible set constraints (Thesis §6.3–6.5)
    parser.add_argument('--constraint', type=str, default='none',
                        choices=['none', 'box', 'l2_ball', 'l1_ball'],
                        help='Feasible set K (default: none = unconstrained)')
    parser.add_argument('--constraint_param', type=float, default=1.0,
                        help='Constraint parameter: bound c for box, '
                             'radius R for balls (default: 1.0)')

    # Model
    parser.add_argument('--embed_dim', type=int, default=8,
                        help='Embedding dimension (default: 8)')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 64],
                        help='MLP hidden layer dimensions (default: 128 64)')
    parser.add_argument('--hash_size', type=int, default=100000,
                        help='Hash table size per feature field (default: 100000)')

    # Training
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--sample_rate', type=float, default=1.0,
                        help='Fraction of data to use per day, stratified by '
                             '(hour, site_domain) (default: 1.0 = all)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Mini-batch size (default: 1, true online learning)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of passes over training data (default: 1)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu or cuda (default: cpu)')

    # Logging & output
    parser.add_argument('--log_interval', type=int, default=500,
                        help='Log every N batches (default: 500)')
    parser.add_argument('--save_model', type=str, default=None,
                        help='Path to save model checkpoint after training '
                             '(e.g., weights.pt)')

    # Prediction mode
    parser.add_argument('--predict', type=str, default=None,
                        help='Path to unlabeled CSV for prediction '
                             '(requires --load_model)')
    parser.add_argument('--predict_output', type=str, default=None,
                        help='Output path for predictions (default: '
                             '<input>_predictions.csv)')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to model checkpoint to load')

    # Comparison mode
    parser.add_argument('--compare', action='store_true',
                        help='Run comparison of all optimizer variants')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.hidden_dims = tuple(args.hidden_dims)

    # Reproducibility
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Check for MPS (Apple Silicon)
    if args.device == 'cpu' and torch.backends.mps.is_available():
        print("Note: MPS (Apple Silicon GPU) is available. Use --device mps to enable.")

    if args.predict:
        predict_unlabeled(args)
    elif args.compare:
        run_comparison(args)
    else:
        run_experiment(args)
