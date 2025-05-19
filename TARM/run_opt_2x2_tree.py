import os
import subprocess
import shutil
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed


def RunOptimization(script, param, log_subdir):
    """
    Start a subprocess to execute the script and redirect stdout/stderr to the corresponding log file.
    """
    log_file = './tmp/' + log_subdir + f"/{param}.log"
    cmd = ["python", script, '--loc', param]
    with open(log_file, 'wb') as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        returncode = proc.wait()
    task_name = f"{script}:{param}"
    return task_name, returncode


def ParallelRun(max_workers=20):
    """
    Parallel execution of Python scripts to optimize 2x2 multiplier and 4-bit adder tree
    """
    TASK_GROUPS = [
        {
            "script": "optimize_2x2.py",
            "params": [''.join(bits) for bits in product('hl', repeat=4)],  # hhhh, hhhl, …, llll
            "log_subdir": "opt_log_2x2"
        },
        {
            "script": "optimize_tree.py",
            "params": [''.join(bits) for bits in product('hl', repeat=2)],  # hh, hl, lh, ll
            "log_subdir": "opt_log_tree"
        }
    ]
    
    os.makedirs('./tmp/', exist_ok=True)
    all_tasks = []

    for group in TASK_GROUPS:
        log_dir = './tmp/' + group["log_subdir"]
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        print(f"Cleared and recreated: {log_dir}")
        script = group["script"]
        log_subdir = group["log_subdir"]
        for p in group["params"]:
            all_tasks.append((script, p, log_subdir))

    total = len(all_tasks)
    print(f"Starting {total} jobs with up to {max_workers} concurrent workers...\n")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(RunOptimization, script, param, log_dir): (script, param)
            for script, param, log_dir in all_tasks
        }
        for future in as_completed(future_to_task):
            script, param = future_to_task[future]
            try:
                task_name, code = future.result()
                status = "OK" if code == 0 else f"ERROR(code {code})"
            except Exception as exc:
                task_name = f"{script}:{param}"
                status = f"ERROR(exception: {exc})"
            print(f"[{task_name}] -> {status}")
            results.append((task_name, status))

    succeeded = sum(1 for _, s in results if s.startswith("OK"))
    print(f"\nDone: {succeeded}/{total} succeeded, {total - succeeded} failed.")


if __name__ == "__main__":
    ParallelRun(max_workers=20)