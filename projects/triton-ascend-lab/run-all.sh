#!/bin/bash
# Triton-Ascend Lab — 一键运行全部 kernel
set -e

KERNELS=(kernel-1-vector-add kernel-2-softmax kernel-3-matmul-tile kernel-4-fused-layernorm)
PASS=0
FAIL=0

for k in "${KERNELS[@]}"; do
    echo "=== $k ==="
    if python3 "$k/kernel.py" 2>&1; then
        echo "  ✅ PASS"
        ((PASS++))
    else
        echo "  ❌ FAIL"
        ((FAIL++))
    fi
    echo ""
done

echo "---"
echo "Results: $PASS passed, $FAIL failed"
