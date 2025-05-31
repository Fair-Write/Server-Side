import os
import time
import errno

class FileLockTimeout(Exception):
    pass

class FileLock:
    def __init__(self, lock_dir, timeout=5, interval=0.1):
        self.lock_dir = lock_dir
        self.timeout = timeout
        self.interval = interval
        self._acquired = False

    def acquire(self):
        start_time = time.time()
        while True:
            try:
                os.mkdir(self.lock_dir)
                self._acquired = True
                return
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                if time.time() - start_time >= self.timeout:
                    raise FileLockTimeout("Could not acquire lock")
                time.sleep(self.interval)

    def release(self):
        if self._acquired:
            try:
                os.rmdir(self.lock_dir)
                self._acquired = False
            except OSError:
                pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

def read_counter(filename: str) -> int:
    try:
        with open(filename, 'r') as f:
            data = f.read().strip()
            if not data:
                return 0
            return int(data)
    except (FileNotFoundError, ValueError):
        return 0

def update_counter(filename: str, delta: int):
    lock_dir = filename + '.lock'
    lock = FileLock(lock_dir)
    with lock:
        current = read_counter(filename)
        new_value = current + delta
        temp_filename = f"{filename}.tmp.{os.getpid()}"
        try:
            with open(temp_filename, 'w') as f:
                f.write(str(new_value))
            os.replace(temp_filename, filename)
        finally:
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except OSError:
                    pass

def increment_counter(filename: str):
    update_counter(filename, 1)

def decrement_counter(filename: str):
    update_counter(filename, -1)


    