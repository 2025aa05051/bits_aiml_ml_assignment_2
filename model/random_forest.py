from sklearn.ensemble import RandomForestClassifier
from datapreperation.config import RANDOM_STATE

def build_model():
    return RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
