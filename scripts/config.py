import torch

# Random shuffle of dataset
SEED_RANGE = 5

# Configure the training process
LEARNING_RATE = 0.00044786967275534047 #1e-3 #0.0033808570389529695
MIN_LEARNING_RATE = 0.00015
WEIGHT_DECAY =  3.295317281553449e-06 #1.1067597759365375e-06
USE_DROP_OUT = False
DROPOUT = 0.18988331206284084 #0.2477008537587833
VARRY_LR = False
SCHEDULED = True
MSE_REDUCTION = "adamw" #"mean"
PATIENCE = 16
BATCH_SIZE = 256
VAL_BATCHSIZE = 256
PRINT_TRAIN = False
PRINT_VAL = False
PRINT_TEST = True

# Image format
RESCALE = True
NUM_ROW = 230
NUM_COLUMN = 230
NUM_EPOCHS = 300    # number of epochs to train for
FILE_EXTENSION = ".png"

# Model config
MODEL_NAME = "resnet50"
USE_PRETRAIN = False
DEVICE = torch.device('cuda') #if torch.cuda.is_available() else torch.device('mps')
EMPTY_CUDA_CACHE = False
print('Running on {}'.format(DEVICE))

# Dataset directory: images and output
OUTPUT_FILE = '../dataset/data/outputs/outputs.txt'
IMG_DIR = '../dataset/data'
SAVE_IMG_DIR = '../dataset_run/BLEVE_30x30'
SAVE_OUTPUT_DIR = '../dataset_run/BLEVE_30x30/outputs/outputs.txt'

# Dataset split
SHUFFLE_TRAIN = True
SHUFFLE_VAL = False
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

ORDER = True
ORDER_METHOD = 1
NUM_WORKERS = 0

# location to save model and plots
SAVED_MODEL_DIR = "../saved_models/resnet50"
SAVED_MODEL_NAME = "resnet50"
SAVED_MODEL_FORMAT = ".pt"

SAVE_PLOTS_EPOCH = 4 # save loss plots after these many epochs (the intervals of saving)
SAVE_MODEL_EPOCH = 4 # save model after these many epochs

# store model's run history
LOG_DIR = "../running_logs/resnet50"


# load model from
LOAD_MODEL_LOCATION = "../saved_models/resnet50/resnet50best_model.pt"

# wandb running config
WANDB_PROJECT_NAME = "CNN_BLEVE_resnet50"


# the size of tensor arrays being displayed
NP_FULL_SIZE = False

# nth epoch at which a checkpoint is saved
SAVED_EPOCH = 100
SAVED_AFTER = 999