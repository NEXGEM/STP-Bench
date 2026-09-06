import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import torch


def setup_paths(output_dir: str, features: Tuple[str, ...] = ('global', 'neighbor', 'target')) -> Dict[str, str]:
    """Return a path map for the output directory, creating only the emb
    subdirectories actually needed (`features`, e.g. from a model's
    feature_type) — a model like StNet that uses raw patches directly
    (feature_type: none) needs none of them at all."""
    paths = {
        'patches':          os.path.join(output_dir, 'patches'),
        'patches_neighbor': os.path.join(output_dir, 'patches', 'neighbor'),
        'adata':            os.path.join(output_dir, 'adata'),
        'emb':              os.path.join(output_dir, 'emb'),
        'emb_global':       os.path.join(output_dir, 'emb', 'global'),
        'emb_neighbor':     os.path.join(output_dir, 'emb', 'neighbor'),
        'emb_target':       os.path.join(output_dir, 'emb', 'target'),
        'pos':              os.path.join(output_dir, 'pos'),
    }
    for feature in features:
        key = f'emb_{feature}'
        if key in paths:
            os.makedirs(paths[key], exist_ok=True)
    return paths


def get_available_gpus() -> List[int]:
    """Return list of visible GPU indices from CUDA_VISIBLE_DEVICES, or all GPUs."""
    if not torch.cuda.is_available():
        return []
    env_val = os.environ.get('CUDA_VISIBLE_DEVICES')
    if env_val is not None:
        return [int(g) for g in env_val.split(',') if g.strip()]
    return list(range(torch.cuda.device_count()))


def split_list_for_gpus(items: List, num_gpus: int) -> List[List]:
    """Round-robin split of items across num_gpus buckets."""
    result: List[List] = [[] for _ in range(num_gpus)]
    for i, item in enumerate(items):
        result[i % num_gpus].append(item)
    return result


def run_command(cmd: List[str], verbose: bool = True,
                cwd: Optional[str] = None) -> Tuple[int, str]:
    """Run a shell command, optionally prefixed by VAR=value, and return (code, output)."""
    env = os.environ.copy()
    if cmd and '=' in cmd[0] and not os.path.exists(cmd[0]):
        var_name, var_value = cmd[0].split('=', 1)
        env[var_name] = var_value
        cmd = cmd[1:]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        env=env,
        cwd=cwd,
    )

    # Real destination for the loop below, resolved once: sys.__stdout__ is
    # the plain-console escape hatch suppress_library_output() (and
    # BenchmarkLogger, which uses the same convention) relies on to stay
    # visible while sys.stdout itself is redirected to /dev/null -- but it
    # is None in some embedded/frozen interpreters, and print(file=None)
    # silently falls back to (suppressed) sys.stdout rather than raising,
    # which would make this subprocess's actual status invisible with no
    # sign anything was swallowed. Falling back to the real sys.stdout
    # (captured before any suppression) is always strictly better than
    # that silent swallow. Neither branch reaches a Jupyter/ipykernel
    # cell's displayed output, since ipykernel's stdout replacement isn't
    # sys.__stdout__ either -- a real fix for that would need to detect
    # the kernel and route through it specifically, out of scope here.
    real_stdout = sys.__stdout__ if sys.__stdout__ is not None else sys.stdout

    output = ""
    for line in process.stdout:
        output += line
        if verbose:
            print(line, end="", file=real_stdout, flush=True)

    return process.wait(), output
