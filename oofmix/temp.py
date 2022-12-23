
from sklearn.model_selection import KFold
import pandas as pd


class SplitObject():
    def __init__(self,n_splits:int=3,random_state:int=0):
        """_summary_

        Parameters
        ----------
        n_splits : int, optional
            _description_, by default 3
        random_state : int, optional
            _description_, by default 0
        """
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X: pd.DataFrame , y: pd.Series , model_list:list = None):
        """_summary_

        Parameters
        ----------
        model_list : list
            _description_
        """
        # The idea is of a single automl object, so we will have to figure out
        # when it is a classification (stratified) and when it is not
        # Also wwhy not stratify in regression using bins???
        skf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        features = X.colunms
        X['oof_predictions'] = 0

        for fold,(train_idx, valid_idx) in enumerate(skf.split(X, y)):
            valid_x = X.loc[valid_idx, features ]
            valid_y = y.loc[valid_idx]

            train_x = X.loc[train_idx, features ]
            train_y = y.loc[train_idx]

            # need to have the individual models in each split saved
            model = 0 # need to figure out

            model.fit(train_x, train_y)

            oof_preds = model.predict(valid_x)

            X.loc[valid_idx, 'oof_predictions']  = model.predict(valid_x)




