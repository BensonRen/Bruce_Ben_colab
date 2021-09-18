"""
Parameter file for specifying the running parameters for forward model
"""
#DATA_SET = 'bruce_high_freq_return'
DATA_SET = 'image'
COMP_IND = 10107

SKIP_CONNECTION = False
USE_CONV = True
SQUARE = True
AVERAGE_EVERY_N = 390
#DIM_X = int(8190 / AVERAGE_EVERY_N)
DIM_X = 3
DIM_Y = 1
LAST_DIM = 128

# The parameter set for linear model
LINEAR = [DIM_X,  DIM_Y]      #Still using two FC layers to have compatiable number of input parameters to tconv layers
CONV_OUT_CHANNEL = []
CONV_KERNEL_SIZE = []
CONV_STRIDE = []

# The parameter set for image model
CHANNEL_LIST = [16, 32 , 64, 128]

# Hyperparameters
OPTIM = "Adam"
REG_SCALE = 0
BATCH_SIZE = 128
EVAL_STEP = 10
TRAIN_STEP = 500
LEARN_RATE = 1e-2
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
