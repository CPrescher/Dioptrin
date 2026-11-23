"""Capture the machine and allocation once, outside the timed benchmark."""

import csv
import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


SLURM_FIELDS = (
    "SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_JOB_NUM_NODES",
    "SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_CPU_BIND_TYPE",
    "SLURM_CPU_BIND_LIST", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU",
    "SLURM_JOB_GPUS", "CUDA_VISIBLE_DEVICES",
)


def command(*args):
    try:
        return subprocess.check_output(
            args, text=True, encoding="utf-8", errors="replace",
            stderr=subprocess.DEVNULL, timeout=8,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def cpu_model():
    if sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                return value.strip()
        except OSError:
            pass
    elif sys.platform == "darwin":
        model = command("sysctl", "-n", "machdep.cpu.brand_string")
        if model:
            return model
    else:
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                key, _, value = line.partition(":")
                if key.strip() == "model name":
                    return value.strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def graphics_inventory():
    text = command(
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    )
    if text:
        results = []
        for name, driver, memory in csv.reader(text.splitlines(), skipinitialspace=True):
            results.append(dict(
                name=name.strip(), driver=driver.strip(),
                memory_mib=int(memory.strip()) if memory.strip().isdigit() else None,
            ))
        return results
    if sys.platform == "darwin":
        text = command("system_profiler", "SPDisplaysDataType", "-json", "-detailLevel", "mini")
        if text:
            try:
                devices = json.loads(text).get("SPDisplaysDataType", [])
                return [dict(name=row["sppci_model"], cores=row.get("sppci_cores"))
                        for row in devices if "sppci_model" in row]
            except json.JSONDecodeError:
                pass
    return []


def capture():
    affinity, physical, memory = None, None, None
    logical = os.cpu_count()
    try:
        import psutil
    except ModuleNotFoundError:
        pass
    else:
        try:
            affinity = psutil.Process().cpu_affinity()
        except (AttributeError, NotImplementedError, psutil.Error):
            pass
        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)
        memory = psutil.virtual_memory().total
    core_types = []
    if sys.platform == "darwin":
        for level in range(2):
            prefix = f"hw.perflevel{level}"
            name = command("sysctl", "-n", f"{prefix}.name")
            cores = command("sysctl", "-n", f"{prefix}.physicalcpu")
            if name and cores and cores.isdigit():
                core_types.append(dict(name=name, physical_cores=int(cores)))
    return dict(
        captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        os=platform.platform(),
        architecture=platform.machine(),
        cpu=cpu_model(),
        physical_cores=physical,
        logical_cpus=logical,
        core_types=core_types,
        cpu_affinity=affinity,
        ram_bytes=memory,
        gpu_inventory=graphics_inventory(),
        allocation={key: os.environ[key] for key in SLURM_FIELDS if key in os.environ},
    )
