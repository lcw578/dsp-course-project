# -*- coding: utf-8 -*-
"""Runner script to capture output from main.py"""
import sys
import os
import io

# Force UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Redirect to file
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_log.txt')
log_file = open(log_path, 'w', encoding='utf-8')

class TeeWriter:
    def __init__(self, *writers):
        self.writers = writers
        self.encoding = getattr(writers[0], 'encoding', 'utf-8')
    def write(self, s):
        for w in self.writers:
            w.write(s)
            w.flush()
    def flush(self):
        for w in self.writers:
            w.flush()
    def isatty(self):
        return False
    def reconfigure(self, **kwargs):
        pass

sys.stdout = TeeWriter(sys.stdout, log_file)
sys.stderr = TeeWriter(sys.stderr, log_file)

try:
    from main import main
    main()
except Exception as e:
    import traceback
    traceback.print_exc()
finally:
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    log_file.close()
