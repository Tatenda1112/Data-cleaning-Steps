import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

mandatory_columns = [
    "loan_id",
    "disbursement_date",
    "expire_date",
    "is_employed",
    "loan_amount",
    "number_of_defaults",
    "outstanding_balance",
    "interest_rate",
    "age",
    "remaining_term",
    "salary",
    "sector",
    "currency",
    "employee_sector",
    "status",
    "loan_status",
]

class MandatoryColumns(BaseEstimator, TransformerMixin):
    def __init__(self, mandatory_columns):
        self.mandatory_columns = mandatory_columns
        self.errors = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.errors = [col for col in self.mandatory_columns if col not in X.columns]
        return X

    def get_errors(self):
        return self.errors

class CheckMissingLoanId(BaseEstimator,TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self,X,y=None):
        return self
    
    def transform(self,X):
        return X.loc[X['loan_id'].isnull()]

class CheckMissingValues(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.errors = pd.DataFrame()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        missing_counts = X.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0]
        self.errors = (missing_counts.to_frame(name="Missing Values")
                       .rename_axis("Column")
                       .reset_index()
                       .sort_values('Missing Values', ascending=False))
        return self.errors


class DateConverter(BaseEstimator, TransformerMixin):
    def __init__(self, date_formats=None):
        self.errors = None
        self.date_formats = date_formats or [
            "%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%Y.%m.%d"]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dates_columns = X.filter(regex="date").columns
        X_temp = X.copy()

        not_converted_dates = (X_temp[dates_columns]
                               .apply(lambda col: pd.to_datetime(col, format=self.date_formats[0], errors='coerce'))
                               .loc[lambda df: df.isnull().any(axis=1)])

        self.errors = not_converted_dates
        return not_converted_dates


class CheckInvalidDates(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.errors = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        invalid_dates = X[X["disbursement_date"] > X["expire_date"]]
        self.errors = invalid_dates
        return invalid_dates

class ConvertedNumeric(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X):
        return self

    def transform(self, X):
        num_columns = [
            "loan_amount",
            "number_of_defaults",
            "outstanding_balance",
            "interest_rate",
            "age",
            "remaining_term",
            "salary",
        ]
        X[num_columns] = X[num_columns].apply(pd.to_numeric, errors="coerce")
        not_converted_num = X.loc[X[num_columns].isnull().any(axis=1)]
        return not_converted_num


class CheckNegativeAmountsAndZeros(BaseEstimator, TransformerMixin):
    def __init__(self, num_columns_ck=None):
        self.num_columns_ck = num_columns_ck or ["loan_amount", "interest_rate", "age", "salary"]
        self.errors = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        mask = X[self.num_columns_ck].apply(lambda col: (col <= 0)).any(axis=1)
        negative_amounts_and_zeros = X[mask]
        return negative_amounts_and_zeros


class CheckDuplicates(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.errors = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        duplicates = X[X.duplicated(keep=False)].sort_values("loan_id")
        return duplicates
