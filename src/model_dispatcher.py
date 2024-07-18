from sklearn import tree
from sklearn import ensemble
import xgboost as xgb
import lightgbm as lgbm
import catboost

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

models = {  "xgb1": xgb.XGBClassifier(**{'alpha': 0.677695836170184, 
                                        'booster': 'gbtree', 
                                        'colsample_bytree': 0.9839666295212105, 
                                        'eta': 0.4038974410191662, 
                                        'gamma': 0.020947918706917257, 
                                        'lambda': 0.7408437454544827, 
                                        'max_depth': 23, 
                                        'min_child_weight': 1, 
                                        'subsample': 0.9327701075232196}),            # plutot pas mal

            "xgb2": xgb.XGBClassifier(**{'alpha': 0.007183123216505892, 
                                        'booster': 'dart', 
                                        'colsample_bytree': 0.6823582309818718, 
                                        'eta': 0.6117661273807745, 
                                        'gamma': 0.4884956017262517, 
                                        'lambda': 0.5120455240717775, 
                                        'max_depth': 10, 
                                        'min_child_weight': 2.0, 
                                        'subsample': 0.9933010350401206}),


            "random_forest": RandomForestClassifier(
                n_estimators=100, 
                criterion="gini", 
                max_depth=None, 
                min_samples_split=2, 
                min_samples_leaf=1, 
                class_weight="balanced",  # Addressing class imbalance
                max_features = None,
                random_state=42),

            "svm": SVC(
                kernel="linear", 
                class_weight="balanced",  # Addressing class imbalance
                random_state=42),

            #"decision_tree_entropy": tree.DecisionTreeClassifier(criterion="entropy"),
            
            "xgb": xgb.XGBClassifier(),

            "hist": ensemble.HistGradientBoostingClassifier(), 

            "extra": ensemble.ExtraTreesClassifier(), 

            "lgbm": lgbm.LGBMClassifier(n_jobs=-1),

            "lgbm1": lgbm.LGBMClassifier(n_jobs=-1, verbose_eval=False, verbose=-1,
                                        **{'bagging_fraction': 0.9890071005455401,
                                        'booster': 'dart', 'class_weight': 'balanced',
                                        'feature_fraction': 0.33930002686209193, 
                                        'lambda_l1': 0.4515515667476831, 
                                        'lambda_l2': 0.8736505526580867, 
                                        'learning_rate': 0.8694864371631662, 
                                        'max_bin': 92, 'max_depth': 6,
                                        'min_data_in_leaf': 4679, 
                                        'min_sum_hessian_in_leaf': 10, 
                                        'num_iterations': 197, 
                                        'num_leaves': 81, 
                                        'subsample': 0.5064014257012379}),

            'gbm': ensemble.GradientBoostingClassifier(),

            'cat': catboost.CatBoostClassifier(verbose=False),

            "dt": tree.DecisionTreeClassifier(criterion="gini", 
                                            max_depth=19, 
                                            min_samples_split=3, 
                                            min_samples_leaf=3)
}

""""logistic_regression": LogisticRegression(
                multi_class="multinomial", 
                max_iter=1000,
                class_weight="balanced",  # Addressing class imbalance
                solver="saga",  # Supports multinomial loss
                random_state=42),"""

""""gradient_boosting": GradientBoostingClassifier(
                n_estimators=100, 
                learning_rate=0.1, 
                max_depth=3, 
                random_state=42),"""


""""neural_network": MLPClassifier(
            hidden_layer_sizes=(100,), 
            activation="relu", 
            solver="adam", 
            alpha=0.0001, 
            batch_size="auto", 
            learning_rate="constant", 
            learning_rate_init=0.001, 
            max_iter=200, 
            shuffle=True, 
            random_state=42),"""



"""{'alpha': 0.015279460125112332, 
'booster': 0, 
'colsample_bytree': 0.7026141965357052, 
'eta': 0.8023635245197862, 
'gamma': 0.19771917651571141, 
'lambda': 0.5945790994229301, 
'max_depth': 17.0, 
'min_child_weight': 1.0, 
'subsample': 0.6323539224252441}"""

"""{'alpha': 0.007183123216505892, 
'booster': 'dart', 
'colsample_bytree': 0.6823582309818718, 
'eta': 0.6117661273807745, 
'gamma': 0.4884956017262517, 
'lambda': 0.5120455240717775, 
'max_depth': 10, 
'min_child_weight': 2.0, 
'subsample': 0.9933010350401206} these params take too much time



"""