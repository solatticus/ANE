#!/bin/bash
# Test 2: Checkpoint round-trip — save, reload, verify continuity
echo "=== TEST 2: Checkpoint Round-Trip ==="
echo ""

cd ~/src/ANE/training

# Clean slate
rm -f ane_lora_ckpt.bin

# Run 20 steps (2 batches), save checkpoint
echo "Phase 1: Training 20 steps from scratch..."
OUTPUT1=$(./train_lora --steps 20 2>/dev/null)
echo "$OUTPUT1" | grep -E "(batch|step [0-9]|Trainable|Report)"

# Record final state
FINAL_BATCH1=$(echo "$OUTPUT1" | grep "batch 2:" | head -1)
FINAL_NORMS1=$(echo "$OUTPUT1" | grep "|A|=" | tail -1)
echo ""
echo "After 20 steps:"
echo "  $FINAL_BATCH1"
echo "  $FINAL_NORMS1"

# Check checkpoint exists
ls -la ane_lora_ckpt.bin
echo ""

# Resume from checkpoint with 10 more steps (1 batch)
echo "Phase 2: Resuming from checkpoint for 10 more steps..."
OUTPUT2=$(./train_lora --steps 30 --resume ane_lora_ckpt.bin 2>/dev/null)
echo "$OUTPUT2" | head -5
echo "..."
echo "$OUTPUT2" | grep -E "(batch|Resumed|step [0-9])"

RESUME_NORMS=$(echo "$OUTPUT2" | grep "|A|=" | head -1)
echo ""
echo "After resume:"
echo "  $RESUME_NORMS"
echo ""

# Key check: resumed |A| and |B| should closely match saved values
echo "PASS criteria: Resumed norms should match saved norms (proving checkpoint preserved weights)"
echo "  Saved:   $FINAL_NORMS1"
echo "  Resumed: $RESUME_NORMS"
