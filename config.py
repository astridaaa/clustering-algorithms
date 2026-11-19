from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
INPUT_FILE = DATA_DIR / "A.csv"
PREPARED_DIR = DATA_DIR / "paruosta"
STABILITY_DIR = PREPARED_DIR / "stabilumas"


CLASS_COLUMN = "klase"
CLASSES = ["N", "L", "R", "V", "A"]
OUTLIER_COLUMN = "ekstremali_isskirtis"

META_COLUMNS = ["id", "irasas", "R", CLASS_COLUMN, "ritmas", OUTLIER_COLUMN]

EXPECTED_OBJECTS = 4837        
EXPECTED_RR_ERRORS = 2        

FEATURE_SETS = {
    "visi": ["RR_sant_vid", "RR_pre", "RR_sant_post", "QRS", "QTc"],
    "ritmo": ["RR_sant_vid", "RR_pre", "RR_sant_post"],
    "formos": ["QRS", "QTc"],
}
FEATURES = FEATURE_SETS["visi"]

MAIN_SET = "visi"


CORRELATION_THRESHOLD = 0.9

RR_VID_ERROR_MS = 5000


SPACES = ["originali", "pca", "tsne"]
N_COMPONENTS = {"pca": 3, "tsne": 2}

TSNE_PARAMS = {"perplexity": 50, "init": "pca"}
TSNE_RANDOM_STATE = 42


SUBSAMPLE_FRACTION = 0.8
N_SUBSAMPLES = {"originali": 20, "pca": 20, "tsne": 10}