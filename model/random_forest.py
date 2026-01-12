from sklearn.ensemble import RandomForestClassifier
from datapreperation.config import RANDOM_STATE

def build_model():
     return RandomForestClassifier(
        n_estimators=250,       # ↓ from 300
        max_depth=12,           # limits tree size
        min_samples_leaf=2,     # reduces overfitting + size
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
