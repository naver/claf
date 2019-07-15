
""" NSML is NAVER SMART MACHINE LEARNING PLATFORM for internal (NAVER Corp)"""

IS_ON_NSML = False
DATASET_PATH = None
SESSION_NAME = ""


try:
    from nsml import *
except ImportError:
    pass
