from multiprocessing import cpu_count


DEBUG = False  # Change this to True to see logs
CPU_COUNT = cpu_count()
TQDM_BAR_FORMAT = '\033[37m{percentage:3.0f}%|{bar:30}\033[37m|[Estimated time remaining: {remaining}]\033[0m'



