#!/usr/bin/env python3
"""
Simple GPU task scheduler that distributes tasks from a file across 2 GPUs.
Executes tasks serially, assigning to the next available GPU.
"""

import subprocess
import sys
import os
import time
import threading
from pathlib import Path
from datetime import datetime

def parse_task_file(task_file):
    """Parse task file, skipping comments and empty lines."""
    tasks = []
    with open(task_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line and not line.startswith('#'):
                tasks.append(line)
    return tasks

def run_task(task, gpu_id, results):
    """Run a single task on specified GPU and record result."""
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    start_time = datetime.now()
    print(f"[GPU {gpu_id}] Starting: {task[:80]}...")
    
    try:
        # Execute task (assuming it's a shell command)
        result = subprocess.run(
            task,
            shell=True,
            capture_output=True,
            text=True,
            env=env,
            check=True
        )
        duration = (datetime.now() - start_time).total_seconds()
        print(f"[GPU {gpu_id}] SUCCESS ({duration:.1f}s): {task[:80]}...")
        results['success'].append((task, gpu_id, duration))
    except subprocess.CalledProcessError as e:
        duration = (datetime.now() - start_time).total_seconds()
        print(f"[GPU {gpu_id}] FAILED ({duration:.1f}s): {task[:80]}...")
        print(f"[GPU {gpu_id}] Error: {e.stderr[:200] if e.stderr else str(e)}")
        results['failed'].append((task, gpu_id, duration, str(e)))
    except Exception as e:
        duration = (datetime.now() - start_time).total_seconds()
        print(f"[GPU {gpu_id}] ERROR ({duration:.1f}s): {task[:80]}...")
        print(f"[GPU {gpu_id}] Exception: {str(e)}")
        results['failed'].append((task, gpu_id, duration, str(e)))

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_tasks_on_gpus.py <task_file>")
        print("Example: python run_tasks_on_gpus.py task_list.txt")
        sys.exit(1)
    
    task_file = Path(sys.argv[1])
    if not task_file.exists():
        print(f"Error: Task file not found: {task_file}")
        sys.exit(1)
    
    tasks = parse_task_file(task_file)
    if not tasks:
        print("No tasks found in file (after filtering comments/empty lines)")
        sys.exit(1)
    
    print(f"Found {len(tasks)} tasks to execute on 2 GPUs")
    print("=" * 80)
    
    results = {'success': [], 'failed': []}
    threads = []
    task_idx = 0
    gpu_available = [True, True]  # Track if GPU 0 and GPU 1 are available
    
    def task_completed_callback(gpu_id):
        """Callback when a task completes on a GPU."""
        gpu_available[gpu_id] = True
    
    # Execute tasks
    while task_idx < len(tasks) or any(not gpu for gpu in gpu_available):
        # Find available GPU
        gpu_id = None
        for i, available in enumerate(gpu_available):
            if available:
                gpu_id = i
                break
        
        if gpu_id is not None and task_idx < len(tasks):
            # Assign task to this GPU
            task = tasks[task_idx]
            gpu_available[gpu_id] = False
            task_idx += 1
            
            # Wait 2 seconds when assigning second task (first to GPU 1) to avoid same-second starts
            if task_idx == 2:
                time.sleep(2)
            
            # Create thread for this task
            def task_wrapper(t, gpu, callback):
                run_task(t, gpu, results)
                callback(gpu)
            
            thread = threading.Thread(
                target=task_wrapper,
                args=(task, gpu_id, task_completed_callback),
                daemon=False
            )
            thread.start()
            threads.append(thread)
        else:
            # Wait a bit if no GPU available
            time.sleep(0.5)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Print summary
    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total tasks: {len(tasks)}")
    print(f"Successful: {len(results['success'])}")
    print(f"Failed: {len(results['failed'])}")
    
    if results['success']:
        print("\n--- SUCCESSFUL TASKS ---")
        for task, gpu, duration in results['success']:
            print(f"[GPU {gpu}] ({duration:.1f}s) {task}")
    
    if results['failed']:
        print("\n--- FAILED TASKS ---")
        for task, gpu, duration, error in results['failed']:
            print(f"[GPU {gpu}] ({duration:.1f}s) {task}")
            print(f"  Error: {error[:100]}")
    
    print("=" * 80)
    
    # Exit with error code if any failed
    sys.exit(1 if results['failed'] else 0)

if __name__ == "__main__":
    main()

