"""
The script for creating result file and testing it
"""

import os

if __name__ == '__main__':
    os.system("python ir_engine.py -s -p -w 'tfidf' -o results.txt")
    os.system('python eval_ir.py cacm_gold_std.txt results.txt')