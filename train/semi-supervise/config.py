# Initialization and configuration
# Hyper params config (Using the best performance on page 11)
alp0 = 1
alp1 = 0.005
alp2 = 0.01
alp3 = 1

ld1 = 1
ld2 = 1

# LR params
mu1 = 2e-4  # For J loss
mu2 = 2e-4  # For L loss

BS = 64
feature_size = 200

# Training epoch
EPCH_START = 980
EPCH_END = 2000

# Device selection
DEV = "cuda"

SAMPLE_INTERVAL = 10
CHCK_PNT_INTERVAL = 50

# Which model want to load?
LOAD_FE = True
LOAD_DIS = True
LOAD_GEN = True
