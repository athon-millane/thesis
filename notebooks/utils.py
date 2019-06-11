# Helper functions #

def size(obj, suffix='B'):
    """
    Get the size of provided object in human readable format.
    """
    import sys
    num = sys.getsizeof(obj)
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def freememory():
    """
    Run garbage collection to free up memory.
    """
    import gc
    gc.collect()