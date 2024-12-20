from typing import Optional

import cvxpy as cp
from cvxpy.constraints.constraint import Constraint
from multiprocessing import Pool
import numpy as np

from carbon_tracing.day_ahead_forecasting.generation_constraints import (
    balance_constraints,
    operation_constraints,
    outage_constraints,
)


class GenerationPostProcessor:
    """
    This class includes methods to generate constraints and solve/
    modify day-ahead generation forecasting outputs using cvxpy.

    Args:
        err_tol_value: the value of the err_tol parameter used in the
            balance_constraints function.
        use_operation_constraints: whether to use operation constraints.
        use_outage_constraints: whether to use outage constraints.
        cp_solver: solver used for optimization with cvxpy. The default
            solver used by cvxpy is "ECOS".

    Attributes:
        use_balance_constraints: whether balance constraints are used.
        use_outage_constraints: whether outage constraints are used.
        err_tol: cp.Parameter that controls the error tolerance in balance
            constraints.
        cp_solver: solver used for optimization with cvxpy.
    """
    def __init__(
        self,
        err_tol_value: int,
        use_operation_constraints: bool,
        use_outage_constraints: bool,
        cp_solver: Optional[str] = None,
    ):
        self.err_tol = cp.Parameter(nonneg=True, value=err_tol_value)
        self.use_operation_constraints = use_operation_constraints
        self.use_outage_constraints = use_outage_constraints
        self.cp_solver = cp_solver

    def generate_constraints(
        self,
        gen_modified: cp.Variable,
        load_total: float,
        line_losses: float,
        gen_minimum: Optional[np.ndarray] = None,
        gen_capacity: Optional[np.ndarray] = None,
        gen_status: Optional[np.ndarray] = None,
    ) -> Constraint:
        """
        Generate Constraint used for solving the optimal modified
        generation forecast at each time step.

        Args:
            gen_modified: modified generation at each bus.
            load_total: total load demand at a time step.
            line_losses: line losses at a time step.
            gen_minimum: minimum amount of energy generation at each bus.
            gen_capacity: maximum amount of energy generation at each bus.
            gen_status: a boolean array indicates the operational status
                of the generators.
        """
        constraints = balance_constraints(
            gen_modified=gen_modified,
            load_total=cp.Constant(load_total),
            line_losses=cp.Constant(line_losses),
            err_tol=self.err_tol,
        )
        if self.use_operation_constraints:
            if gen_minimum is not None and gen_capacity is not None:
                constraints += operation_constraints(
                    gen_modified=gen_modified,
                    gen_minimum=cp.Constant(gen_minimum),
                    gen_capacity=cp.Constant(gen_capacity),
                    gen_status=cp.Constant(gen_status),
                )
            else:
                raise ValueError("gen_minimum or gen_capacity not provided.")
        if self.use_outage_constraints:
            if gen_status is not None:
                constraints += outage_constraints(
                    gen_modified=gen_modified,
                    gen_status=cp.Constant(gen_status),
                )
            else:
                raise ValueError("gen_status is not provided.")
        return constraints

    def solve_modified_gen(
        self,
        gen_forecast: np.ndarray,
        load_total: float,
        line_losses: float,
        gen_minimum: Optional[np.ndarray] = None,
        gen_capacity: Optional[np.ndarray] = None,
        gen_status: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Solve for the optimal modified generation forecast at each time step.

        Args:
            gen_forecast: forecasted generation at a time step.
            load_total: total load demand at a time step.
            line_losses: line losses at a time step.
            gen_minimum: minimum amount of energy generation at each bus.
            gen_capacity: maximum amount of energy generation at each bus.
            gen_status: a boolean array indicates the operational status
                of the generators.
        """
        num_buses = len(gen_forecast)
        gen_modified = cp.Variable(num_buses)
        constraints = self.generate_constraints(
            gen_modified=gen_modified,
            load_total=load_total,
            line_losses=line_losses,
            gen_minimum=gen_minimum,
            gen_capacity=gen_capacity,
            gen_status=gen_status,
        )
        problem = cp.Problem(
            cp.Minimize(cp.sum(cp.abs(gen_modified - gen_forecast))),
            constraints,
        )
        problem.solve(solver=self.cp_solver)
        if problem.status not in ["infeasible", "unbounded"]:
            return gen_modified.value
        else:
            return gen_forecast


def multiprocess_gen_forecast(
    gen_forecast_ts: np.ndarray,
    load_total_ts: np.ndarray,
    line_losses_ts: np.ndarray,
    err_tol_value: int,
    use_operation_constraints: bool,
    use_outage_constraints: bool,
    gen_minimum_ts: Optional[np.ndarray] = None,
    gen_capacity_ts: Optional[np.ndarray] = None,
    gen_status_ts: Optional[np.ndarray] = None,
    cp_solver: Optional[str] = None,
    n_processes: Optional[int] = 1,
) -> np.ndarray:
    """
    Solve for the optimal modified generation forecast at all time steps.
    Arrays gen_forecast_ts, gen_minimum_ts, gen_capacity_ts, and
    gen_status_ts cover data for T timesteps and B buses.

    Args:
        gen_forecast_ts: a T x B array representing forecasted generation
            at all timesteps.
        load_total_ts: a T x 1 array representing total load demand at all
            timesteps.
        line_losses_ts: a T x 1 array representing line losses at all
            timesteps.
        gen_minimum_ts: a T x B array representing minimum amount of energy
            generation at all timesteps.
        gen_capacity_ts: a T x B array representing maximum amount of energy
            generation at all timesteps.
        gen_status_ts: a T x B boolean array indicates the operational status
            of the generators at all timesteps.
        err_tol_value: the value of the err_tol parameter used in the
            balance_constraints function.
        use_operation_constraints: whether to use operation constraints.
        use_outage_constraints: whether to use outage constraints.
        cp_solver: solver used for optimization with cvxpy.
        n_processes: number of CPUs to distribute the calculation.
    """
    timesteps = gen_forecast_ts.shape[0]
    if not use_operation_constraints:
        gen_minimum_ts = np.full(timesteps, None)
        gen_capacity_ts = np.full(timesteps, None)
    if not use_outage_constraints:
        gen_status_ts = np.full(timesteps, None)
    gen_postprocessor = GenerationPostProcessor(
        err_tol_value=err_tol_value,
        use_operation_constraints=use_operation_constraints,
        use_outage_constraints=use_outage_constraints,
        cp_solver=cp_solver,
    )
    with Pool(processes=n_processes) as pool:
        return np.array(
            pool.starmap(
                gen_postprocessor.solve_modified_gen,
                zip(
                    gen_forecast_ts,
                    load_total_ts,
                    line_losses_ts,
                    gen_minimum_ts,
                    gen_capacity_ts,
                    gen_status_ts,
                )
            )
        )
