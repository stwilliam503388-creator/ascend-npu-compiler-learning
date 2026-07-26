#!/bin/bash
# check-docs.sh — lightweight repository consistency checks with no LLVM dependency.

# Intentionally omit `set -e`: this script aggregates all failures before exiting.
# `-u` is safe because variables are initialized before use; `pipefail` propagates
# pipeline failures for detection without forcing immediate script exit.
set -u
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 1
FAIL=0
EXPECTED_BASIC_MLIR_CASES=10
EXPECTED_INTERMEDIATE_MLIR_CASES=11
EXPECTED_ADVANCED_MLIR_CASES=10
EXPECTED_TOTAL_MLIR_CASES=31
EXPECTED_TRITON_CASES=28

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

fail() { echo -e "${RED}✗${NC} $1"; FAIL=$((FAIL + 1)); }
pass() { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC} $1"; }

check_required_paths() {
    local paths=(
        README.md
        SUMMARY.md
        setup.sh
        docs/quickstart.md
        docs/why-ascend.md
        docs/primer/README.md
        docs/llvm/README.md
        docs/mlir/README.md
        docs/ascend/README.md
        docs/ascendnpu-ir/README.md
        projects/README.md
        projects/hello-pass
        projects/opt-pass
        projects/mlir-hello
        projects/toy-mini
        projects/standalone-mlir
        projects/ascendnpu-ir-op-counter
        projects/ascend-samples
        projects/ascendnpu-ir-demo
        projects/triton-ascend-lab
        projects/triton-ascend-lab/run-all.sh
        projects/ascendnpu-ir-demo/run-tests.sh
        projects/ascendnpu-ir-demo/test-cases/mlir/01_basic
        projects/ascendnpu-ir-demo/test-cases/mlir/02_intermediate
        projects/ascendnpu-ir-demo/test-cases/mlir/03_advanced
    )

    for rel in "${paths[@]}"; do
        if [ ! -e "$ROOT/$rel" ]; then fail "missing required path: $rel"; fi
    done
    [ $FAIL -eq 0 ] && pass "required paths exist"
}

check_markdown_links() {
    if python3 - "$ROOT" <<'PYCHECK'
import os, re, sys
from pathlib import Path
root = Path(sys.argv[1])
errors = []
link_re = re.compile(r'\[[^\]]+\]\(([^)]+)\)')
for md in root.rglob('*.md'):
    if '.git' in md.parts or 'plans' in md.parts:
        continue
    text = md.read_text(encoding='utf-8', errors='ignore')
    for match in link_re.finditer(text):
        raw = match.group(1).strip()
        if not raw or raw.startswith(('#', 'http://', 'https://', 'mailto:')):
            continue
        if ' ' in raw and not raw.startswith(('./', '../')):
            continue
        target = raw.split('#', 1)[0]
        if not target:
            continue
        target = target.replace('%20', ' ')
        path = (md.parent / target).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError:
            errors.append(f'{md.relative_to(root)} -> {raw} escapes repository')
            continue
        if not path.exists():
            errors.append(f'{md.relative_to(root)} -> {raw}')
if errors:
    print('\n'.join(errors))
    sys.exit(1)
PYCHECK
    then
        pass "markdown local links resolve"
    else
        fail "markdown local links have missing targets"
    fi
}

check_mlir_cases() {
    local mlir_dir="$ROOT/projects/ascendnpu-ir-demo/test-cases/mlir"
    local total basic intermediate advanced missing_run
    basic=$(find "$mlir_dir/01_basic" -maxdepth 1 -name '*.mlir' | wc -l | tr -d ' ')
    intermediate=$(find "$mlir_dir/02_intermediate" -maxdepth 1 -name '*.mlir' | wc -l | tr -d ' ')
    advanced=$(find "$mlir_dir/03_advanced" -maxdepth 1 -name '*.mlir' | wc -l | tr -d ' ')
    total=$((basic + intermediate + advanced))

    [ "$basic" -eq "$EXPECTED_BASIC_MLIR_CASES" ] || fail "expected $EXPECTED_BASIC_MLIR_CASES basic MLIR cases, got $basic"
    [ "$intermediate" -eq "$EXPECTED_INTERMEDIATE_MLIR_CASES" ] || fail "expected $EXPECTED_INTERMEDIATE_MLIR_CASES intermediate MLIR cases, got $intermediate"
    [ "$advanced" -eq "$EXPECTED_ADVANCED_MLIR_CASES" ] || fail "expected $EXPECTED_ADVANCED_MLIR_CASES advanced MLIR cases, got $advanced"
    [ "$total" -eq "$EXPECTED_TOTAL_MLIR_CASES" ] || fail "expected $EXPECTED_TOTAL_MLIR_CASES total MLIR cases, got $total"

    missing_run=""
    while IFS= read -r mlir; do
        if ! grep -q "^// RUN:" "$mlir"; then
            missing_run="${missing_run}${mlir}\n"
        fi
    done < <(find "$mlir_dir" -mindepth 2 -maxdepth 2 -name '*.mlir' | sort)
    if [ -n "$missing_run" ]; then
        printf '%b' "$missing_run"
        fail "some MLIR cases do not contain RUN annotations"
    fi

    local triton_total
    triton_total=$(find "$ROOT/projects/ascendnpu-ir-demo/test-cases/triton" -mindepth 2 -maxdepth 2 -name '*.py' | wc -l | tr -d ' ')
    [ "$triton_total" -eq "$EXPECTED_TRITON_CASES" ] || fail "expected $EXPECTED_TRITON_CASES Triton cases, got $triton_total"

    if [ $FAIL -eq 0 ]; then pass "MLIR/Triton case counts and RUN annotations are consistent"; fi
}

check_shell_scripts() {
    local scripts=(setup.sh scripts/check-docs.sh projects/ascendnpu-ir-demo/run-tests.sh projects/ascendnpu-ir-demo/run-demo.sh)
    for rel in "${scripts[@]}"; do
        bash -n "$ROOT/$rel" || fail "shell syntax failed: $rel"
    done
    [ $FAIL -eq 0 ] && pass "shell script syntax checks pass"
}

echo "╔════════════════════════════════════════════╗"
echo "║  ascend-npu-compiler-learning: 文档检查    ║"
echo "╚════════════════════════════════════════════╝"

check_required_paths
n_before=$FAIL
check_markdown_links
[ $FAIL -gt $n_before ] && warn "上方列出了缺失链接，请修复路径或改为纯文本"
check_mlir_cases
check_shell_scripts

if [ $FAIL -gt 0 ]; then
    echo ""
    echo -e "${RED}检查失败: $FAIL 项${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}全部检查通过${NC}"
