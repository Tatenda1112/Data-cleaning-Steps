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
    A custom transformer that ensures specified mandatory columns are present in a pandas DataFrame.

    Parameters
    ----------
    mandatory_columns : list
        A list of column names that are required to be present in the DataFrame.

    Attributes
    ----------
    mandatory_columns : list
        Stores the list of mandatory column names.
    errors : list or None
        A list of missing mandatory columns if any are not found in the DataFrame,
        otherwise an empty list. Initially set to None.

    Methods
    -------
    fit(X, y=None)
        Fits the transformer, returning itself. This method doesn't alter the DataFrame.

    transform(X)
        Checks if all mandatory columns are present in the DataFrame. If any are missing,
        stores them in `self.errors`. Returns the unaltered DataFrame `X`.

    get_errors()
        Returns the list of missing mandatory columns.
    """

    def __init__(self, mandatory_columns):
        self.mandatory_columns = mandatory_columns
        self.errors = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        missing_columns = [
            col for col in self.mandatory_columns if col not in X.columns
        ]
        if missing_columns:
            self.errors = missing_columns
        else:
            self.errors = []
        return X

    def get_errors(self):
        return self.errors


class CheckMissingValues(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer that checks for missing values in a DataFrame.

    This transformer identifies columns with missing values and provides a summary of the number of missing values in each column.

    Attributes:
    ----------
    errors : pd.DataFrame
        A DataFrame containing the names of columns with missing values and the count of missing values in each column.

    Methods:
    -------
    fit(X, y=None):
        Fits the transformer. This method is a placeholder and does not perform any fitting.

    transform(X):
        Identifies columns with missing values in the DataFrame and stores the results in the 'errors' attribute.
        Returns a DataFrame with the names of the columns and the corresponding count of missing values.
    """

    def __init__(self):
        """
        Initializes the CheckMissingValues transformer.
        """
        self.errors = pd.DataFrame()

    def fit(self, X, y=None):
        """
        Fits the transformer. This method is a placeholder and does not perform any fitting.

        Parameters:
        ----------
        X : pd.DataFrame
            The input DataFrame to fit the transformer on.
        y : None
            An optional parameter, not used in this method.

        Returns:
        -------
        self : CheckMissingValues
            Returns the instance itself.
        """
        return self

    def transform(self, X):
        """
        Identifies columns with missing values in the DataFrame and stores the results.

        Parameters:
        ----------
        X : pd.DataFrame
            The input DataFrame in which missing values are to be checked.

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing the names of the columns with missing values and the count of missing values in each column.
            Also stores this information in the 'errors' attribute for inspection.
        """
        missing_counts = X.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0]
        self.errors = pd.DataFrame(
            {"Column": missing_counts.index, "Missing Values": missing_counts.values}
        ).sort_values('Missing Values',ascending=False)

        return self.errors


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
        self.errors = not_converted_dates

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


class CheckInvalidDates(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer that identifies rows with invalid date ranges in a DataFrame.

    This transformer is used to find rows where the 'disbursement_date' is greater than the 'expire_date'.
    It is assumed that the date columns are already in datetime format.

    Attributes:
    ----------
    errors : pd.DataFrame, optional
        A DataFrame containing rows with invalid dates, where 'disbursement_date' > 'expire_date'.

    Methods:
    -------
    fit(X, y=None):
        Fits the transformer. This method is a placeholder and does not perform any fitting.

    transform(X):
        Identifies rows with invalid date ranges in the DataFrame.
        Returns a DataFrame with rows where 'disbursement_date' is greater than 'expire_date'.
    """

    def __init__(self):
        """
        Initializes the CheckInvalidDates transformer.
        """
        self.errors = None

    def fit(self, X, y=None):
        """
        Fits the transformer. This method is a placeholder and does not perform any fitting.

        Parameters:
        ----------
        X : pd.DataFrame
            The input DataFrame to fit the transformer on.
        y : None
            An optional parameter, not used in this method.

        Returns:
        -------
        self : CheckInvalidDates
            Returns the instance itself.
        """
        return self

    def transform(self, X):
        """
        Identifies rows with invalid date ranges in the DataFrame.

        Parameters:
        ----------
        X : pd.DataFrame
            The input DataFrame with 'disbursement_date' and 'expire_date' columns in datetime format.

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing rows where 'disbursement_date' is greater than 'expire_date'.
            Also stores these rows in the 'errors' attribute for inspection.
        """

        dates_columns = X[X["disbursement_date"] > X["expire_date"]]
        self.errors = dates_columns

        return dates_columns


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
