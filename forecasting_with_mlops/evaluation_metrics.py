"""
A collection of metrics that can be used to evaluate the accuracy and
validity of the day-ahead forecasting and power flow (based on forecasted
loads and generations) results.
"""
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats


def carbon_intensity_corr(
    ci_pred: np.ndarray,
    ci_true: np.ndarray,
    corr: str,
    average: Optional[bool] = True,
) -> Union[float, np.ndarray]:
    """
    Calculate the rank correlation using either Spearman's rho
    or Kendall's tau. This metric is used for quantifying the
    ordinal association between predicted and true carbon intensities
    within a power system network. The datatype of input variables is
    subject to change once the tracing output data schema is defined.

    Args:
        ci_pred: an array with a shape of num_of_timesteps X
            num_of_buses representing the carbon intensities based
            on the forecasted load and generation.
        ci_true: an array with a shape of num_of_timesteps X
            num_of_buses representing the carbon intensities based
            on the true load and generation.
        corr: correlation statistics used for calculating the rank
            correlations. Supported options include spearmanr and
            kendalltau.
        average: whether to average the rank correlations across
            the timesteps.
    """
    if corr == "spearmanr":
        corr_func = np.vectorize(
            stats.spearmanr,
            signature="(n),(n)->(),()"
        )
    elif corr == "kendalltau":
        corr_func = np.vectorize(
            stats.kendalltau,
            signature="(n),(n)->(),()"
        )
    else:
        raise ValueError("corr must be spearmanr or kendalltau")
    if average:
        return corr_func(ci_pred, ci_true)[0].mean()
    else:
        return corr_func(ci_pred, ci_true)[0]


def power_flow_diff(
    pf_from_pred: pd.DataFrame,
    pf_to_pred: pd.DataFrame,
    pf_from_true: pd.DataFrame,
    pf_to_true: pd.DataFrame,
    line_weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    This function calculate the mean squared errors (or weighted squared
    errors), which is used to evaluate the power flow solutions based
    on the forecasted load and generation.
    PFSolutionProcessor._process_powerflow_ts_df() method can be used to
    prepare the input pf_from and pf_to. But the datatype of input variables
    is subject to change once the power flow data schema is defined.

    Args:
        pf_from_pred: a dataframe of true power flow value from a node,
            prepared by the PFSolutionProcessor._process_powerflow_ts_df()
            method.
        pf_to_pred: a dataframe of true power flow value to a node.
        pf_from_true: a dataframe of predicted power flow value from a node.
        pf_to_true: a dataframe of predicted power flow value to a node.
        line_weights: a dictionary indicates the weight of individual lines,
            this may be determined via sensitivity factors from the power
            flow analysis.
    """
    num_lines = len(pf_from_pred.columns)
    pf_true = (
        pd.concat([pf_from_true, pf_to_true]).groupby(level=0).mean()
    )
    pf_pred = (
        pd.concat([pf_from_pred, pf_to_pred]).groupby(level=0).mean()
    )
    if line_weights is not None:
        return (
            ((pf_true-pf_pred) ** 2).dot(pd.Series(line_weights))
            / pd.Series(line_weights).sum()
        )
    else:
        return ((pf_pred - pf_true) ** 2).sum(axis=1) / num_lines
