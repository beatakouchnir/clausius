"""Run an mlx-lm LoRA arm under a hard memory ceiling.

A first attempt peaked at **128.3 GB on a 128 GB machine** — close enough to an
OOM to take the whole system down. MLX defaults its memory limit to 1.5x the
recommended working set, which on this hardware means it will happily push into
swap and thrash rather than fail. That default is wrong for an unattended
training run: a job that dies with an exception costs a restart, a job that
exhausts system memory costs the machine.

So this wrapper sets an explicit ceiling BEFORE importing anything that
allocates. Over the limit, MLX raises instead of swapping.

The configs it drives also carry the fixes that brought real usage down:
gradient checkpointing on, batch 2 instead of 4, sequence length 160 instead of
256 (documents are ~110 tokens, so 256 was pure headroom), and a small
validation batch count — the 25-batch validation pass was itself a peak.

Usage:
  python3 -m knowledge.finetune --config records/corpus/arms/router.yaml
  python3 -m knowledge.finetune --config ... --limit-gb 90 --iters 6
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--config', required=True)
    ap.add_argument('--limit-gb', type=float, default=95.0,
                    help='hard ceiling; over it MLX raises rather than swaps')
    ap.add_argument('--iters', type=int, default=None)
    ap.add_argument('--adapter-path', default=None)
    a, rest = ap.parse_known_args()

    import mlx.core as mx
    prev = mx.set_memory_limit(int(a.limit_gb * 1024 ** 3))
    print(f"memory limit {a.limit_gb:.0f} GB (was {prev / 1024 ** 3:.0f} GB)",
          flush=True)

    argv = ['mlx_lm.lora', '-c', a.config]
    if a.iters is not None:
        argv += ['--iters', str(a.iters)]
    if a.adapter_path:
        argv += ['--adapter-path', a.adapter_path]
    argv += rest
    sys.argv = argv

    from mlx_lm.lora import main as lora_main
    lora_main()


if __name__ == '__main__':
    main()
