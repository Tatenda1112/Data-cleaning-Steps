import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class DateConverter(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer for converting date columns in a DataFrame to datetime objects.

    This transformer attempts to convert date columns to datetime objects using a list of specified date formats.
    If conversion fails for all provided formats, it returns the rows with dates that could not be converted.

    Parameters
    ----------
    date_formats : list of str, optional
        A list of date formats to try for conversion. Each format should be a string compatible with `pd.to_datetime()`.
        If None, a default list of formats will be used:
        ["%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%Y.%m.%d"].

    Attributes
    ----------
    errors : str
        Stores error messages if any exceptions occur during the conversion process.

    Methods
    -------
    fit(X, y=None)
        Does nothing and simply returns the instance. This is a placeholder method for compatibility with scikit-learn.

    transform(X)
        Converts date columns in the DataFrame `X` to datetime objects using the specified formats.
        Returns a DataFrame containing rows with dates that could not be converted if any format fails.
    """

    def __init__(self, date_formats=None):
        """
        Initializes the DateConverter with specified date formats.

        Parameters
        ----------
        date_formats : list of str, optional
            A list of date formats to try for conversion. Each format should be a string compatible with `pd.to_datetime()`.
            If None, a default list of formats will be used:
            ["%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%Y.%m.%d"].
        """
        self.errors = None
        self.date_formats = (
            date_formats
            if date_formats is not None
            else ["%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%Y.%m.%d"]
        )

    def fit(self, X, y=None):
        """
        Placeholder method for compatibility with scikit-learn. Does nothing and simply returns the instance.

        Parameters
        ----------
        X : pandas DataFrame
            The input DataFrame (not used).
        y : None, optional
            The target values (not used).

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        return self

    def transform(self, X):
        """
        Converts date columns in the DataFrame `X` to datetime objects using the specified date formats.

        Parameters
        ----------
        X : pandas DataFrame
            The input DataFrame with date columns to be converted.

        Returns
        -------
        pandas DataFrame
            A DataFrame containing rows with dates that could not be converted if any format fails. If all dates are successfully
            converted, it returns an empty DataFrame.
        """
        dates_columns = X.filter(regex="date").columns
        X_temp = X.copy()
        not_converted_dates = pd.DataFrame()

        for date_format in self.date_formats:
            try:
                X_temp[dates_columns] = X_temp[dates_columns].apply(
                    pd.to_datetime, format=date_format, errors="coerce"
                )
            except Exception:
                continue
            if X_temp[dates_columns].isnull().any().any():
                not_converted_dates = X_temp[X_temp[dates_columns].isnull().any(axis=1)]
                if not not_converted_dates.empty:
                    break
            else:
                break

        return not_converted_dates


class ConvertedNumeric(BaseEstimator, TransformerMixin):
    """
    A custom transformer for converting specified columns in a DataFrame to numeric types.

    Attributes:
        None
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        Fit method for the transformer. In this case, it does nothing but is required for compatibility with scikit-learn.

        Parameters:
            X (pd.DataFrame): The input data.
            y (None): Ignored. Not used for this transformer.

        Returns:
            self: The instance of the transformer.
        """
        return self

    def transform(self, X):
        """
        Transform the data by converting specified columns to numeric types and identifying rows with failed conversions.

        Parameters:
            X (pd.DataFrame): The input data.

        Returns:
            pd.DataFrame: DataFrame containing rows with columns that could not be converted to numeric.
        """
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
    A custom transformer for checking for negative amounts and zeros in specified numeric columns.

    Attributes:
        num_columns_ck (list): List of columns to check for negative values and zeros.
        errors (None): Placeholder for storing errors.
    """

    def __init__(self, num_columns_ck=None):
        if num_columns_ck is None:
            num_columns_ck = ["loan_amount", "interest_rate", "age", "salary"]
        self.num_columns_ck = num_columns_ck
        self.errors = None

    def fit(self, X, y=None):
        """
        Fit method for the transformer. In this case, it does nothing but is required for compatibility with scikit-learn.

        Parameters:
            X (pd.DataFrame): The input data.
            y (None): Ignored. Not used for this transformer.

        Returns:
            self: The instance of the transformer.
        """
        return self

    def transform(self, X):
        """
        Transform the data by identifying rows with negative values or zeros in specified numeric columns.

        Parameters:
            X (pd.DataFrame): The input data.

        Returns:
            pd.DataFrame: DataFrame containing rows with negative values or zeros in the specified columns.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError
        missing_cols = [col for col in self.num_columns_ck if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {', '.join(missing_cols)}")
        conditions = [(X[col] < 0) | (X[col] == 0) for col in self.num_columns_ck]
        mask = pd.concat(conditions, axis=1).any(axis=1)
        negative_amounts_and_zeros = X[mask]
        return negative_amounts_and_zeros


class CheckDuplicates(BaseEstimator, TransformerMixin):
    """
    A custom transformer for checking for duplicate rows in a DataFrame based on the 'loan_id' column.

    Attributes:
        errors (None): Placeholder for storing errors.
    """

    def __init__(self):
        self.errors = None

    def fit(self, X, y=None):
        """
        Fit method for the transformer. In this case, it does nothing but is required for compatibility with scikit-learn.

        Parameters:
            X (pd.DataFrame): The input data.
            y (None): Ignored. Not used for this transformer.

        Returns:
            self: The instance of the transformer.
        """
        return self

    def transform(self, X):
        """
        Transform the data by identifying duplicate rows based on the 'loan_id' column.

        Parameters:
            X (pd.DataFrame): The input data.

        Returns:
            pd.DataFrame: DataFrame containing duplicate rows sorted by 'loan_id'.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a DataFrame")
        duplicates = X.loc[X.duplicated(keep=False)].sort_values("loan_id")
        return duplicates
