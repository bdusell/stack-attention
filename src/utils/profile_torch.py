import dataclasses
import datetime

import torch

@dataclasses.dataclass
class ProfileResult:
    duration: float
    memory_allocated: int
    memory_reserved: int
    initial_memory_stats: dict
    memory_stats: dict

def profile(func, device, warmup=True):
    torch.cuda.synchronize(device)
    initial_memory_stats = torch.cuda.memory_stats(device)
    if warmup:
        func()
        torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start_time = datetime.datetime.now()
    func()
    torch.cuda.synchronize(device)
    duration = (datetime.datetime.now() - start_time).total_seconds()
    memory_stats = torch.cuda.memory_stats(device)
    memory_allocated = (
        memory_stats['allocated_bytes.all.peak'] -
        initial_memory_stats['allocated_bytes.all.current']
    )
    memory_reserved = memory_stats['reserved_bytes.all.peak']
    return ProfileResult(
        duration,
        memory_allocated,
        memory_reserved,
        initial_memory_stats,
        memory_stats
    )

def get_current_memory(device):
    torch.cuda.synchronize(device)
    return torch.cuda.memory_stats(device)['allocated_bytes.all.current']

def reset_memory_profiler(device):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

def get_peak_memory(device):
    torch.cuda.synchronize(device)
    return torch.cuda.memory_stats(device)['allocated_bytes.all.peak']
