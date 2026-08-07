"""
verify_checkpoint.py — Quick sanity check that best_gatv2.pt loads correctly
after re-zipping. Run this BEFORE starting the FastAPI server.

Usage:
    python verify_checkpoint.py path/to/best_gatv2.pt
"""
import sys
import torch

def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_checkpoint.py path/to/best_gatv2.pt")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Loading: {path}")

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"\n❌ FAILED to load — the .pt file is likely corrupted or not a valid zip.")
        print(f"   Error: {e}")
        print("   → Re-do the re-zip step, make sure you're zipping the FILES")
        print("     inside the folder (data.pkl, data/, version, etc.), not the")
        print("     folder itself, and use standard ZIP compression.")
        sys.exit(1)

    print("\n✅ Loaded successfully. Checkpoint contents:")
    for k, v in ckpt.items():
        if k == "model_state":
            print(f"  - model_state: {len(v)} tensors")
        elif k == "class2idx":
            print(f"  - class2idx: {v}")
        else:
            print(f"  - {k}: {v if not isinstance(v, dict) else '(dict)'}")

if __name__ == "__main__":
    main()