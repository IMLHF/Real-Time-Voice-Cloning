import gc
import resource
import psutil
import os

def prt_mem_used(name):
    process = psutil.Process(os.getpid())
    print('%15s Used Memory:' % name, process.memory_info().rss / 1024 / 1024,'MB', flush=True)

# def prt_mem_used(name):
#     gc.collect()
#     max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
#     print("%15s_mem used: %dMB" % (name, max_mem_used / 1024), flush=True)