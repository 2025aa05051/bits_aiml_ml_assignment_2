from xgboost import XGBClassifier
from datapreperation.config import RANDOM_STATE

def build_model():
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )
