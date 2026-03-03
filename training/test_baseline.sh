#!/bin/bash
# Test 3: Baseline throughput comparison — train_large vs train_lora timing
echo "=== TEST 3: Throughput Comparison ==="
echo "train_large (full training) vs train_lora (LoRA fine-tuning)"
echo ""

cd ~/src/ANE/training

# Run train_large for 10 steps (1 batch)
echo "Running train_large --steps 10..."
LARGE_OUT=$(./train_large --steps 10 2>/dev/null)
LARGE_WALL=$(echo "$LARGE_OUT" | grep "Wall time:" | awk '{print $3}')
LARGE_COMPILE=$(echo "$LARGE_OUT" | grep "Compile time:" | awk '{print $3}')
LARGE_MS=$(echo "$LARGE_OUT" | grep "ms/step\|train=" | tail -1)
echo "$LARGE_OUT" | grep -E "(Wall time|Compile time|ms/step|batch)"
echo ""

# Run train_lora for 10 steps (1 batch), fresh
rm -f ane_lora_ckpt.bin
echo "Running train_lora --steps 10..."
LORA_OUT=$(./train_lora --steps 10 2>/dev/null)
LORA_WALL=$(echo "$LORA_OUT" | grep "Wall time:" | awk '{print $3}')
LORA_COMPILE=$(echo "$LORA_OUT" | grep "Compile time:" | awk '{print $3}')
LORA_MS=$(echo "$LORA_OUT" | grep "ms/step\|train=" | tail -1)
echo "$LORA_OUT" | grep -E "(Wall time|Compile time|ms/step|batch)"
echo ""

echo "=== Summary ==="
echo "train_large: wall=$LARGE_WALL compile=$LARGE_COMPILE"
echo "train_lora:  wall=$LORA_WALL compile=$LORA_COMPILE"
echo ""
echo "Key insight: LoRA adds CPU-side merge cost (~1ms per layer for 768x768 sgemm)"
echo "but the dominant cost is ANE kernel compilation (~3s per batch)."
echo "Per-step train time should be nearly identical since the ANE kernels are the same."
