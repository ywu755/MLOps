"""
This script includes a list of constraints can be used for modifying
day-ahead generation forecasting outputs. We use this approach as
a proxy without solving the full unit commitment/economic dispatch.

"""
from typing import List, Optional

import cvxpy as cp
from cvxpy.constraints.constraint import Constraint


def balance_constraints(
    gen_modified: cp.Variable,
    load_total: cp.Constant,
    line_losses: cp.Constant,
    err_tol: cp.Parameter,
) -> List[Constraint]:
    """
    Generate constraints for balancing generation and load at a time step.

    Args:
        gen_modified: modified generation at each bus.
        load_total: total load demand at a time step.
        line_losses: line losses at a time step.
        err_tol: a scaling parameter that controls the error tolerance based
            on the multiple of the line losses. If the losses are based on a
            planning model (which is typically what's provided to us), they
            should be at their highest. Therefore, we shouldn't allow too much
            of a difference with generation being larger than the load.
    """
    gen_load_difference = cp.abs(cp.sum(gen_modified) - load_total)
    return [gen_load_difference <= err_tol * line_losses]


def operation_constraints(
    gen_modified: cp.Variable,
    gen_minimum: cp.Constant,
    gen_capacity: cp.Constant,
    gen_status: Optional[cp.Constant] = None,
) -> List[Constraint]:
    """
    Generate constraints for the operation limits for each generator.

    Args:
        gen_modified: modified generation at each bus.
        gen_minimum: minimum amount of energy generation at each bus.
        gen_capacity: maximum amount of energy generation at each bus.
        gen_status: a boolean array indicates the operational status
            of the generators. One indicates an operating generator
            and zero indicates a generator with scheduled outage.
    """
    if gen_status is not None:
        gen_minimum = cp.multiply(gen_minimum, gen_status)
    return [gen_modified <= gen_capacity, gen_modified >= gen_minimum]


def outage_constraints(
    gen_modified: cp.Variable,
    gen_status: cp.Constant,
) -> List[Constraint]:
    """
    Generate constraints to represent generator outage schedule.

    Args:
        gen_modified: modified generation at each bus.
        gen_status: a boolean array indicates the operational status
            of the generators. One indicates an operating generator
            and zero indicates a generator with scheduled outage.
    """
    return [gen_modified == cp.multiply(gen_modified, gen_status)]
