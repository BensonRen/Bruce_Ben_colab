"""
Parameter file for specifying the running parameters for forward model
"""
DATA_SET = 'bruce_high_freq_return'
COMP_IND = 10693

SKIP_CONNECTION = False
USE_CONV = False
DIM_X = 8190
DIM_Y = 1

LINEAR = [DIM_X, 50, 50, 50, 50, DIM_Y]      #Still using two FC layers to have compatiable number of input parameters to tconv layers
CONV_OUT_CHANNEL = []
CONV_KERNEL_SIZE = []
CONV_STRIDE = []

# Hyperparameters
OPTIM = "Adam"
REG_SCALE = 1e-2
BATCH_SIZE = 128
EVAL_STEP = 2
TRAIN_STEP = 100
LEARN_RATE = 1e-3
LR_DECAY_RATE = 0.2
STOP_THRESHOLD = 1e-9
DROPOUT = 0
SKIP_HEAD = 3
SKIP_TAIL = [2, 4, 6] #Currently useless

FORCE_RUN = True
TEST_RATIO = 0.2
RAND_SEED = 1

# Running specific
USE_CPU_ONLY = False
MODEL_NAME  = None
EVAL_MODEL = None
NUM_COM_PLOT_TENSORBOARD = 1
