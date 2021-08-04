# Device
DEV = "cpu"

# Training epoch
EPCH_START = 1000
EPCH = 2000

# Batch size
BS = 8

# Learning Rate
LR_DEFAULT = 0.0001
LR_D_EEG = LR_DEFAULT
LR_G_EEG = LR_DEFAULT
LR_D_IMG = LR_DEFAULT
LR_G_IMG = LR_DEFAULT

# Beta value
B1 = 0.5
B2 = 0.999

DECAY_EPCH = 100

LAMBDA_CYC = 10.0
# LAMBDA_ID = 5.0 # Since our approach is impossible, I will comment this out

SAMPLE_INTERVAL = 100
CHCK_PNT_INTERVAL = 100
