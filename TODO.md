# TODO: 重启后继续 Power Benchmark

## 背景
长时间连续 benchmark + CoreML ANE 编译导致 GPU 状态不稳定（GPU Hang Error），需要重启恢复。

## 重启前准备
- [x] 确保所有代码已 commit 并 push（commit 638f4fb）
- [x] asitop 需要重启后重新启动

## 重启后步骤

### 0. 启动 asitop（必须先做）
```bash
sudo asitop
```
保持这个终端窗口开着，PowerMonitor 会自动发现 `/tmp/asitop_powermetrics*` 文件。

### 1. 重跑 0.8B hybrid_ane long（之前 GPU hang 崩溃）
```bash
source 3-11-mlx-community-env/bin/activate
cd /Users/alex/Documents/mlx-community/HybridInference2
PYTHONUNBUFFERED=1 python3 tests/benchmark.py \
    --backends hybrid_ane --models 0.8B \
    --prompt-lengths long --num-runs 4 --append
```
**验证点**：原始数据 TTFT=100ms, decode=73.3 tok/s。上次跑出 892ms/67.0 tok/s，需确认是否为系统退化。

### 2. 重跑 2B-bf16 hybrid_ane（之前 3 次 GPU page fault）
```bash
PYTHONUNBUFFERED=1 python3 tests/benchmark.py \
    --backends hybrid_ane --models 2B-bf16 \
    --prompt-lengths short medium long --num-runs 4 --append
```
**验证点**：原始数据 TTFT=22/54/122ms, decode=100-104 tok/s。上次跑出 34 tok/s + GPU crash。

### 3. 对比新旧数据
跑完后检查 `results/benchmark_results.json` 里带 `power_*` 字段的新数据：
- 如果 TTFT 和 decode 恢复正常 → 说明之前是 GPU 过热/状态退化
- 如果仍然异常 → 说明 chunked CoreML 本身有性能问题

### 4. 更新文档（如果数据有变化）
- `results/table_for_paper.md` — 更新 power 数据
- `docs/paper.tex` — 更新 Table 8/9
- `docs/index.html` — 更新 power 表格
- `docs/paper.pdf` — `cd docs && tectonic paper.tex`
- Git commit + push

## 已完成的 Power 数据（无需重跑）

| Config | CPU (W) | GPU (W) | ANE (W) | Status |
|--------|---------|---------|---------|--------|
| 0.8B baseline short | 9.5 | 6.7 | 0 | OK |
| 0.8B baseline medium | 7.0 | 18.0 | 0 | OK |
| 0.8B baseline long | 6.7 | 19.0 | 0 | OK |
| 0.8B hybrid short | 7.4 | 14.6 | 0.024 | OK |
| 0.8B hybrid medium | 10.5 | 5.2 | 0.002 | OK |
| **0.8B hybrid long** | 10.4 | 6.2 | 0.017 | **需重跑验证** |
| 2B-8bit baseline short | 9.0 | 21.2 | 0 | OK |
| 2B-8bit baseline medium | 8.4 | 25.8 | 0 | OK |
| 2B-8bit baseline long | 8.7 | 30.9 | 0 | OK |
| 2B-bf16 baseline short | 8.8 | 19.3 | 0 | OK |
| 2B-bf16 baseline medium | 8.5 | 21.3 | 0 | OK |
| 2B-bf16 baseline long | 7.9 | 23.7 | 0 | OK |
| **2B-bf16 hybrid** | — | — | — | **需重跑（GPU crash）** |
| 9B-8bit baseline short | 6.6 | 36.5 | 0 | OK |
| 9B-8bit baseline medium | 6.2 | 41.6 | 0 | OK |
| 9B-8bit baseline long | 6.3 | 46.9 | 0 | OK |
| 9B-8bit hybrid short | 8.1 | 31.3 | 0 | OK |
| 9B-8bit hybrid medium | 9.5 | 21.5 | 0 | OK |
| 9B-8bit hybrid long | 11.1 | 11.4 | 0 | OK |

## 关键文件
- `tests/benchmark.py` — 集成了 PowerMonitor（自动采集功耗）
- `tests/parse_power.py` — PowerMonitor 类（读 asitop 的 powermetrics plist）
- `results/benchmark_results.json` — 所有 benchmark 数据（含 power_* 字段）
