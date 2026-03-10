
#configuration du project

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

#FOLDERS directories

#data directories
DATA_DIR = BASE_DIR / "data"

# Clean data directories
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Raw data directorie
RAW_DATA_DIR = DATA_DIR / "raw"

#Model directorie
MODELS_DIR = BASE_DIR / "models"

#artefact directorie
ARTEFACTS_DIR = MODELS_DIR / "artifacts"

# FILES DIRECTORIES

RAW_DATA_FILE = RAW_DATA_DIR / "FuelConsumptionCo2.csv"
MODEL_FILE = MODELS_DIR / "m_l_regression.pkl"
SCALES_FILE = ARTEFACTS_DIR / "scale.pkl"

X_TEST_FILE = PROCESSED_DATA_DIR / "X_test.csv"
Y_TEST_FILE = PROCESSED_DATA_DIR / "Y_test.csv"
X_TEST_TRANSFORM = PROCESSED_DATA_DIR / "X_test_transform.csv"
X_TRAIN_SCALED = PROCESSED_DATA_DIR / "X_train_scaled.csv"
X_TRAIN_FILE   = ARTEFACTS_DIR / "X_train.csv"
Y_TRAIN_FILE   = ARTEFACTS_DIR / "y_train.csv"
FINAL_DF       = ARTEFACTS_DIR / "final_df.csv"


# fonfiguration du model
TEST_SIZE = 0.2
RANDOM_STATE = 42

#features

FEATURES_TO_USE_3 = ['ENGINESIZE','FUELCONSUMPTION_COMB_MPG']
FEATURES_TO_USE_2 = ['ENGINESIZE']
FEATURES_TO_USE_1 = ['FUELCONSUMPTION_COMB_MPG']
FEATURES_TO_USE_4 = ['FUELCONSUMPTION_COMB']
FEATURES_TO_USE_5 = ['CYLINDERS']
FEATURES_TO_USE_6 = ['FUELCONSUMPTION_CITY']

# Target
TARGET_VARIABLE = "CO2EMISSIONS"





