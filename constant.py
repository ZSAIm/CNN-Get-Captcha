
CHAR_SET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
            'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
            'u', 'v', 'w', 'x',  'y', 'z', '0', '1', '2', '3',
            '4', '5', '6', '7', '8']

MODEL_SAVE_PATH = 'model\\'
MODEL_NAME = 'model.ckpt'

DATA_SET_PATH = 'dataset\\'
TFRECORD_NAME = 'data.tfrecord-%.5d-of-%.5d'

IMAGE_WIDTH = 72
IMAGE_HEIGHT = 27
NUM_CHANNELS = 1

NUM_CAPTCHA = 4
LABEL_LEN = len(CHAR_SET)

INPUT_NODE = IMAGE_WIDTH * IMAGE_HEIGHT
OUTPUT_NODE = NUM_CAPTCHA * LABEL_LEN

BATCH_SIZE = 100

TRAINING_STEP = 10000



