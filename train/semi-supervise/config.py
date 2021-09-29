# Initialization and configuration
# Hyper params config (Using the best performance on page 11)
alp0 = 1
alp1 = 0.005
alp2 = 0.01
alp3 = 1

ld1 = 1
ld2 = 1

LAMBDA_GP = 10

# LR params
mu1 = 2e-4  # For J loss
mu2 = 5e-5  # For L loss

BS = 10
BS_UNPAIRED = 64
feature_size = 200

# Training epoch
EPCH_START = 0
EPCH_END = 6000

# Device selection
DEV = "cuda:1"

CHCK_PNT_INTERVAL = 100
SAMPLE_INTERVAL = 10

# Which model want to load?
LOAD_FE = True
LOAD_DIS = True
LOAD_GEN = True

# Prevent gradient explode
MAX_GRAD_FLOAT32 = 3e+38
MIN_GRAD_FLOAT32 = -3e+38

# Model parameters clamping
WEIGHT_MIN = -0.01
WEIGHT_MAX = -WEIGHT_MIN

# WGANs needs to train discriminator more
DIS_TRAIN_ITER = 5
