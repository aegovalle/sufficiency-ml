"""Consolidation of the gate, training part (Draft v0.2, Section 20): three seeds per model,
GeM exponent with its own learning rate, AE with d in {8, 32}. Existing seed-0 GAP and AE-d8
models are re-embedded on the extended corpus (test2) instead of retrained.

Run:  .venv\\Scripts\\python scripts\\run_consolidation_train.py [--epochs 60]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from suffml.train.train_gate import train_supervised, train_ae, reembed_supervised, reembed_ae

ROOT = os.path.join(os.path.dirname(__file__), "..")
MD = os.path.join(ROOT, "results", "models")
SEEDS = (0, 1, 2)
GAP = ("R_E", "RH_b20")
GEM = ("R_E", "RH_b20", "RH_b2", "RH_b40")


def done(tag):
    return os.path.exists(os.path.join(MD, f"{tag}_emb.npz")) and os.path.exists(os.path.join(MD, f"{tag}.consol"))


def mark(tag):
    open(os.path.join(MD, f"{tag}.consol"), "w").write("ok")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=60); a = ap.parse_args()
    for t in GAP:
        for s in SEEDS:
            tag = f"sup_{t}_gap_d8_s{s}"
            if done(tag):
                continue
            if s == 0 and os.path.exists(os.path.join(MD, f"{tag}.pt")):
                reembed_supervised(tag)
            else:
                train_supervised(target=t, pool="gap", d=8, Lambda=16.0, epochs=a.epochs, seed=s, tag=tag)
            mark(tag)
    for d in (8, 32):
        for s in SEEDS:
            tag = f"ae_d{d}_s{s}"
            if done(tag):
                continue
            if d == 8 and s == 0 and os.path.exists(os.path.join(MD, f"{tag}.pt")):
                reembed_ae(tag)
            else:
                train_ae(d=d, epochs=a.epochs, seed=s, tag=tag)
            mark(tag)
    for t in GEM:
        for s in SEEDS:
            tag = f"sup_{t}_gem2_d8_s{s}"
            if done(tag):
                continue
            train_supervised(target=t, pool="gem", d=8, Lambda=16.0, epochs=a.epochs, seed=s, tag=tag, lr_q=0.05)
            mark(tag)
    print("training done")


if __name__ == "__main__":
    main()
