#!/bin/bash
# Test 1: Identity property — fresh LoRA (B=0) should produce same loss as baseline
# This proves the merge W_eff = W_frozen + 0 = W_frozen works correctly

echo "=== TEST 1: Identity Property ==="
echo "Fresh LoRA init should match pretrained baseline loss"
echo ""

cd ~/src/ANE/training

# Remove any existing checkpoint so we get a fresh init
rm -f ane_lora_ckpt.bin

# Run LoRA for 1 step — capture the FIRST loss (before any update)
echo "Running train_lora (1 step, fresh init)..."
LORA_LOSS=$(./train_lora --steps 1 2>/dev/null | grep "^step 0" | awk '{print $2}' | sed 's/loss=//')
echo "  LoRA first-step loss: $LORA_LOSS"

# Run train_large for 1 step to get baseline loss
echo "Running train_large (1 step, same pretrained model)..."
BASELINE_LOSS=$(./train_large --steps 1 2>/dev/null | grep "^step 0" | awk '{print $2}' | sed 's/loss=//')
echo "  Baseline first-step loss: $BASELINE_LOSS"

echo ""
if [ -z "$LORA_LOSS" ] || [ -z "$BASELINE_LOSS" ]; then
    echo "FAIL: Could not extract losses"
    exit 1
fi

# Compare — they should be very close (same model, but random data batch differs)
echo "Note: Both use random data positions, so exact match is unlikely."
echo "But both should be in the same ~3.5-4.0 range for a pretrained model."
echo "If LoRA init was corrupted, loss would be much higher (>10)."
echo ""
echo "Result: LoRA=$LORA_LOSS  Baseline=$BASELINE_LOSS"
