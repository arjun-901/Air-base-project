DATA_PATH = "data/air_quality.csv"
DATETIME_COL = "datetime"

FEATURES = ['PM2.5','PM10','NO2','SO2','CO','O3','temp','humidity','wind']
TARGET = 'AQI'

WINDOW = 24
PRED_HORIZON = 1

IMG_SIZE_VGG9 = (64, 64)
IMG_SIZE_VGG16 = (224, 224)

EPOCHS = 50
BATCH_SIZE = 32
PATIENCE = 8

MODEL_DIR = "models"

RANDOM_SEED = 42

