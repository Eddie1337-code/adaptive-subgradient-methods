"""
Hyperparameter Tuning for AdaGrad COMID and RDA.

Systematic grid search following the principled approach:
  Step 1: Tune learning rate η (most important, no regularization)
  Step 2: Tune regularization strength λ (with best η from step 1)
  Step 3: Tune constraint parameter R (with best η and λ)

Tuning is performed on days 1–2 (training) with evaluation on day 3,
following the online learning protocol. Final experiments use all days.

Results are saved to JSON for reproducibility and analysis.
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import random
import torch
import torch.nn as nn

from data import detect_features, iterate_minibatches
from model import CTRModel
from train import get_labeled_files, evaluate


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tuning_results')


def tune_single(optimizer_name, lr, reg_type, reg_lambda, constraint,
                constraint_param, train_files, eval_file, features,
                n_fields, args):
    """Train on train_files, evaluate on eval_file. Return metrics."""
    from optimizers import DiagonalAdaGradCOMID, DiagonalAdaGradRDA

    # Fresh model for each run
    torch.manual_seed(args.seed)
    model = CTRModel(
        n_fields=n_fields,
        hash_size=args.hash_size,
        embed_dim=args.embed_dim,
        hidden_dims=args.hidden_dims,
    ).to(args.device)

    # Build optimizer
    if optimizer_name == 'comid':
        optimizer = DiagonalAdaGradCOMID(
            model.parameters(), lr=lr, delta=args.delta,
            reg_type=reg_type, reg_lambda=reg_lambda,
            constraint=constraint, constraint_param=constraint_param)
    elif optimizer_name == 'rda':
        optimizer = DiagonalAdaGradRDA(
            model.parameters(), lr=lr, delta=args.delta,
            reg_type=reg_type, reg_lambda=reg_lambda,
            constraint=constraint, constraint_param=constraint_param)
    elif optimizer_name == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, eps=args.delta)

    criterion = nn.BCEWithLogitsLoss()

    # Train
    model.train()
    running_loss = torch.tensor(0.0, device=args.device)
    n_samples = 0
    for f in train_files:
        for indices, labels in iterate_minibatches(
            f, features, batch_size=args.batch_size,
            hash_size=args.hash_size, device=args.device,
            shuffle_chunks=False, sample_rate=args.sample_rate
        ):
            optimizer.zero_grad(set_to_none=True)
            logits = model(indices)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.detach() * len(labels)
            n_samples += len(labels)

    train_loss = running_loss.item() / max(n_samples, 1)

    # Evaluate
    results = evaluate(model, criterion, eval_file, features, args)
    results['train_loss'] = train_loss
    results['train_samples'] = n_samples

    # Sparsity
    total_params = sum(p.numel() for p in model.parameters())
    total_zeros = sum((p.data == 0).sum().item() for p in model.parameters())
    results['sparsity'] = total_zeros / total_params

    return results


def run_grid_search(grid, optimizer_name, train_files, eval_file,
                    features, n_fields, args, step_name):
    """Run grid search, return sorted results."""
    results = []
    total = len(grid)

    print(f"\n{'='*70}")
    print(f"{step_name}: {optimizer_name.upper()} — {total} configurations")
    print(f"{'='*70}")

    for i, config in enumerate(grid):
        t0 = time.time()
        metrics = tune_single(
            optimizer_name=optimizer_name,
            lr=config['lr'],
            reg_type=config.get('reg', 'none'),
            reg_lambda=config.get('reg_lambda', 0.0),
            constraint=config.get('constraint', 'none'),
            constraint_param=config.get('constraint_param', 1.0),
            train_files=train_files,
            eval_file=eval_file,
            features=features,
            n_fields=n_fields,
            args=args,
        )
        elapsed = time.time() - t0

        entry = {**config, **metrics, 'time': elapsed}
        results.append(entry)

        desc = ', '.join(f'{k}={v}' for k, v in config.items())
        print(f"  [{i+1}/{total}] {desc}  "
              f"→ AUC={metrics['auc']:.6f}  "
              f"LogLoss={metrics['logloss']:.6f}  "
              f"Sparsity={metrics['sparsity']:.1%}  "
              f"({elapsed:.1f}s)")

    # Sort by AUC descending
    results.sort(key=lambda x: -x['auc'])

    print(f"\n  Best: AUC={results[0]['auc']:.6f}  "
          f"({', '.join(f'{k}={results[0][k]}' for k in config.keys())})")

    return results


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--sample_rate', type=float, default=0.1)
    parser.add_argument('--hash_size', type=int, default=5000)
    parser.add_argument('--embed_dim', type=int, default=4)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[32, 16])
    parser.add_argument('--delta', type=float, default=1e-10)
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path (default: tuning_results/<timestamp>.json)')
    args = parser.parse_args()
    args.hidden_dims = tuple(args.hidden_dims)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Tuning data: train on days 1–2, evaluate on day 3
    labeled = get_labeled_files()
    train_files = labeled[:2]
    eval_file = labeled[2]
    features, n_fields, _ = detect_features(train_files[0])

    print(f"Tuning setup:")
    print(f"  Train: {[os.path.basename(f) for f in train_files]}")
    print(f"  Eval:  {os.path.basename(eval_file)}")
    print(f"  Features: {n_fields} fields")
    print(f"  Sample rate: {args.sample_rate:.0%}")
    print(f"  Batch size: {args.batch_size}")

    all_results = {}

    # ── Step 1: Tune learning rate η ──
    lr_grid = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

    for opt in ['comid', 'rda', 'adagrad']:
        grid = [{'lr': lr} for lr in lr_grid]
        results = run_grid_search(
            grid, opt, train_files, eval_file,
            features, n_fields, args,
            step_name=f"Step 1: Learning Rate")
        all_results[f'step1_lr_{opt}'] = results

    best_lr_comid = all_results['step1_lr_comid'][0]['lr']
    best_lr_rda = all_results['step1_lr_rda'][0]['lr']
    best_lr_adagrad = all_results['step1_lr_adagrad'][0]['lr']

    print(f"\n  Best LRs: COMID={best_lr_comid}, RDA={best_lr_rda}, "
          f"AdaGrad={best_lr_adagrad}")

    # ── Step 2: Tune regularization λ ──
    lambda_grid = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

    for opt, best_lr in [('comid', best_lr_comid), ('rda', best_lr_rda)]:
        for reg in ['l1', 'l2']:
            grid = [{'lr': best_lr, 'reg': reg, 'reg_lambda': lam}
                    for lam in lambda_grid]
            results = run_grid_search(
                grid, opt, train_files, eval_file,
                features, n_fields, args,
                step_name=f"Step 2: {reg.upper()} Regularization")
            all_results[f'step2_{reg}_{opt}'] = results

    # ── Step 3: Tune constraint parameter ──
    for opt, best_lr in [('comid', best_lr_comid), ('rda', best_lr_rda)]:
        # Box constraints
        grid = [{'lr': best_lr, 'constraint': 'box', 'constraint_param': c}
                for c in [0.5, 1.0, 2.0, 5.0, 10.0]]
        results = run_grid_search(
            grid, opt, train_files, eval_file,
            features, n_fields, args,
            step_name=f"Step 3: Box Constraint")
        all_results[f'step3_box_{opt}'] = results

        # L2 ball
        grid = [{'lr': best_lr, 'constraint': 'l2_ball', 'constraint_param': r}
                for r in [1.0, 5.0, 10.0, 50.0, 100.0]]
        results = run_grid_search(
            grid, opt, train_files, eval_file,
            features, n_fields, args,
            step_name=f"Step 3: L2-Ball Constraint")
        all_results[f'step3_l2ball_{opt}'] = results

        # L1 ball
        grid = [{'lr': best_lr, 'constraint': 'l1_ball', 'constraint_param': r}
                for r in [1.0, 5.0, 10.0, 50.0, 100.0]]
        results = run_grid_search(
            grid, opt, train_files, eval_file,
            features, n_fields, args,
            step_name=f"Step 3: L1-Ball Constraint")
        all_results[f'step3_l1ball_{opt}'] = results

    # ── Save results ──
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = args.output or os.path.join(
        RESULTS_DIR,
        f"tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    output = {
        'timestamp': datetime.now().isoformat(),
        'args': {k: v for k, v in vars(args).items()
                 if not k.startswith('_')},
        'best_hyperparameters': {
            'comid': {'lr': best_lr_comid},
            'rda': {'lr': best_lr_rda},
            'adagrad': {'lr': best_lr_adagrad},
        },
        'results': all_results,
    }

    # Add best λ per reg type
    for opt in ['comid', 'rda']:
        for reg in ['l1', 'l2']:
            key = f'step2_{reg}_{opt}'
            if key in all_results and all_results[key]:
                best = all_results[key][0]
                output['best_hyperparameters'][opt][f'lambda_{reg}'] = best['reg_lambda']

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("TUNING COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {out_path}")
    print(f"\nBest hyperparameters:")
    for opt, hp in output['best_hyperparameters'].items():
        print(f"  {opt.upper()}: {hp}")


if __name__ == '__main__':
    main()
