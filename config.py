import os

NUM_PROCS = os.cpu_count()

# configure global file paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
GRAPHS_DIR = os.path.join(PROJECT_ROOT, 'graphs')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

DATA_OUT_DIR = os.path.join(DATA_DIR, 'out_out')
IMAGENET64_DIR = os.path.join(DATA_DIR, 'imagenet64')
OXFORD_DIR = os.path.join(DATA_DIR, 'oxford')

GRAPHS_OUT_DIR = os.path.join(GRAPHS_DIR, 'out')

UTILS_DIR = os.path.join(SRC_DIR, 'utils')
