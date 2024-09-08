"""
Data Quality Check Pipeline for Loan Data

This module defines various custom transformers for checking data quality issues
in loan datasets. The transformers are used to perform tasks such as checking for
missing values, invalid dates, negative or zero amounts, duplicates, and ensuring 
the presence of mandatory columns.

Classes:
    - MandatoryColumns: Checks if all mandatory columns are present in the DataFrame.
    - CheckMissingLoanId: Identifies rows with missing loan IDs.
    - CheckMissingValues: Identifies columns with missing values and returns a report.
    - DateConverter: Converts date columns to datetime format and reports unconverted dates.
    - CheckInvalidDates: Identifies rows where disbursement_date is greater than expire_date.
    - ConvertedNumeric: Converts specified columns to numeric types and identifies unconverted rows.
    - CheckNegativeAmountsAndZeros: Checks for negative and zero values in specified numeric columns.
    - CheckDuplicates: Identifies duplicate rows based on loan_id.

Dependencies:
    - pandas
    - numpy
    - sklearn (BaseEstimator, TransformerMixin, Pipeline)
"""

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
    """
    Transformer to check if the DataFrame contains all mandatory columns.

    Attributes:
        - mandatory_columns: List of mandatory columns.
        - errors: List of missing mandatory columns.
    """

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


class CheckMissingLoanId(BaseEstimator, TransformerMixin):
    """
    Transformer to identify rows with missing loan IDs.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.loc[X["loan_id"].isnull()]


class CheckMissingValues(BaseEstimator, TransformerMixin):
    """
    Transformer to identify columns with missing values and return a summary report.

    Attributes:
        - errors: DataFrame containing columns and their respective missing value counts.
    """

    def __init__(self):
        self.errors = pd.DataFrame()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        missing_counts = X.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0]
        self.errors = (
            missing_counts.to_frame(name="Missing Values")
            .rename_axis("Column")
            .reset_index()
            .sort_values("Missing Values", ascending=False)
        )
        return self.errors


class DateConverter(BaseEstimator, TransformerMixin):
    """
    Transformer to convert date columns to datetime format using specified formats.

    Attributes:
        - date_formats: List of date formats to try during conversion.
        - errors: DataFrame containing rows where date conversion failed.
    """

    def __init__(self, date_formats=None):
        self.errors = None
        self.date_formats = date_formats or [
            "%d/%m/%Y",
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d-%m-%Y",
            "%Y.%m.%d",
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dates_columns = X.filter(regex="date").columns
        X_temp = X.copy()

        not_converted_dates = (
            X_temp[dates_columns]
            .apply(
                lambda col: pd.to_datetime(
                    col, format=self.date_formats[0], errors="coerce"
                )
            )
            .loc[lambda df: df.isnull().any(axis=1)]
        )

        self.errors = not_converted_dates
        return not_converted_dates


class CheckInvalidDates(BaseEstimator, TransformerMixin):
    """
    Transformer to check for invalid dates where 'disbursement_date' is later than 'expire_date'.

    Attributes:
        - errors: DataFrame containing rows with invalid dates.
    """

    def __init__(self):
        self.errors = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        invalid_dates = X[X["disbursement_date"] > X["expire_date"]]
        self.errors = invalid_dates
        return invalid_dates


class ConvertedNumeric(BaseEstimator, TransformerMixin):
    """
    Transformer to convert specified columns to numeric types.

    Attributes:
        - num_columns: List of columns to convert to numeric types.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
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
    """
    Transformer to check for negative or zero values in specified numeric columns.

    Attributes:
        - num_columns_ck: List of columns to check for negative or zero values.
        - errors: DataFrame containing rows with negative or zero values.
    """

    def __init__(self, num_columns_ck=None):
        self.num_columns_ck = num_columns_ck or [
            "loan_amount",
            "interest_rate",
            "age",
            "salary",
        ]
        self.errors = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        mask = X[self.num_columns_ck].apply(lambda col: (col <= 0)).any(axis=1)
        negative_amounts_and_zeros = X[mask]
        return negative_amounts_and_zeros


class CheckDuplicates(BaseEstimator, TransformerMixin):
    """
    Transformer to identify duplicate rows based on the 'loan_id' column.

    Attributes:
        - errors: DataFrame containing duplicate rows.
    """

    def __init__(self):
        self.errors = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        duplicates = X[X.duplicated(keep=False)].sort_values("loan_id")
        return duplicates
