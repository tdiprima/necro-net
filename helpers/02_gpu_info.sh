#!/bin/bash
# ⚡️ Can your GPU use AMP safely?
# 7.0+ → AMP is 🔥 (Volta, Turing, Ampere, Ada)
# < 7.0 → AMP might be meh or unstable
# If your GPU is modern → AMP = free speed

nvidia-smi --query-gpu=name,compute_cap --format=csv
