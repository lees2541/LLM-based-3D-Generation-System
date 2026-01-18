# generator/progress_manager.py

import threading

current_progress = 0
generation_lock = threading.Lock()

def global_progress_update(percent):
    global current_progress
    with generation_lock:
        current_progress = percent
