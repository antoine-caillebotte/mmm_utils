"""Utility functions for MMM optimization."""

from typing import Callable


from xarray import DataTree
from xarray import DataArray
import arviz as az
from pymc.model.core import Model

import pymc as pm
from pymc.pytensorf import rvs_in_graph

import pytensor.tensor as pt
from pytensor.graph.basic import Variable
from pytensor.graph.replace import clone_replace
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.graph.traversal import ancestors
from pytensor.xtensor.vectorization import vectorize_graph
from pytensor.compile import Function, function as compile_function
from pytensor.xtensor.type import as_xtensor, xtensor_constant, xtensor


def extract_response_distribution(
    pymc_model: Model,
    idata: DataTree,
    response_variable: str,
):
    """Extract the response distribution from a PyMC model and .
    Rewrites the graph setting the response variable as the output, and replacing
    all free RVs in the graph with xtensor constants containing their posterior samples.

    Parameters
    ----------
    pymc_model : Model
        The PyMC model containing the response variable.
    idata : DataTree
         object containing posterior samples.
    response_variable : str
        Name of the response variable in the model.

    Returns
    -------
    Variable
        The extracted response distribution as a PyTensor variable.

    Raises
    ------
    RuntimeError
        If RVs are found in the extracted graph after processing, which indicates a bug in the
    """

    # Convert  to a sample-major xarray
    posterior = az.extract(idata).transpose("sample", ...)  # type: ignore

    # The PyMC variable to extract
    response_var = pymc_model[response_variable]

    # Identify which free RVs are needed to compute `response_var`.
    # Frozen deterministics are treated as additional blockers so their
    # subgraphs are not traversed — their posterior values are substituted
    # directly, just like free RVs.
    free_rvs = set(pymc_model.free_RVs)
    frozen_vars: set = set()
    # if frozen_deterministics:
    #     for name in frozen_deterministics:
    #         if name in pymc_model.named_vars:
    #             frozen_vars.add(pymc_model[name])

    # We want to find all ancestors of `response_var` that are in `free_rvs` or `frozen_vars`,
    # but we don't want to traverse through any of those variables.
    # So we treat them as blockers in the graph traversal.
    blockers = free_rvs | frozen_vars
    needed_rvs = [
        rv for rv in ancestors([response_var], blockers=blockers) if rv in blockers
    ]

    # Clone the graph, replacing needed RVs with placeholders.
    # This allows us to clean up the graph without modifying the original model's graph,
    # and without worrying about accidentally messing with the original RVs
    #  (which are still in the model's graph) when we do cleanup rewrites.
    placeholder_replace_dict = {pymc_model[rv.name]: rv.clone() for rv in needed_rvs}

    [response_var] = clone_replace(
        [response_var],
        replace=placeholder_replace_dict,
    )

    if rvs_in_graph([response_var]):
        raise RuntimeError("RVs found in the extracted graph, this is likely a bug")

    # Cleanup graph :
    # * canonicalize : rewrite the graph to a canonical form, which helps with subsequent rewrites
    # * ShapeOpt : remove unnecessary shape manipulations that can get in the way of vectorization
    response_var = rewrite_graph(response_var, include=("canonicalize", "ShapeOpt"))

    # Replace placeholders with actual posterior samples
    # 1. Extract posterior samples for needed RVs, convert to xtensor constants,
    # and create a replace dict
    replace_dict = {}
    for placeholder in placeholder_replace_dict.values():
        replace_dict[placeholder] = xtensor_constant(
            posterior[placeholder.name].astype(placeholder.dtype),
            name=placeholder.name,
        )

    # 2. Vectorize across samples & replace placeholders with sample-major xtensor constants

    print("✅ Replacing RVs with posterior samples in the graph:")
    for placeholder, constant in replace_dict.items():
        print(f"\t * {placeholder} \t with constant {constant}")

    response_distribution = vectorize_graph(response_var, replace=replace_dict)

    # Final cleanup
    response_distribution = rewrite_graph(
        response_distribution,
        include=(
            "useless",
            "local_eager_useless_unbatched_blockwise",
            "local_useless_unbatched_blockwise",
        ),
    )

    return response_distribution


def replace_variable_by_optimization_variable(
    pymc_model, name, xr_data: DataArray, extra_replacements: dict | None = None
):
    """Replace a variable in the PyMC model graph with an optimization variable.

    Parameters
    ----------
    pymc_model : Model
        The PyMC model containing the variable to replace.
    name : str
        The name of the variable to replace.
    xr_data : xarray.DataArray
        The xarray DataArray containing the data for the variable,
        used to determine the shape and dimensions of the optimization variable.
    extra_replacements : dict, optional
        Additional replacements to apply to the model graph, by default None.
    Returns
    -------
    tuple
        A tuple containing the optimization variable (as an xtensor)
        and the PyTensor graph of the model with the variable replaced.
    """
    input_flat = xtensor(
        name=f"{name}_flat",
        shape=(xr_data.size,),
        dims=(f"{name}_flat",),
    )

    input_variable = as_xtensor(
        pt.reshape(input_flat.values, xr_data.shape),  # pylint: disable=E1101, no-member
        dims=xr_data.dims,
        name=name,
    )

    replacements = {name: input_variable}
    if extra_replacements:
        replacements.update(extra_replacements)

    return input_flat, pm.do(
        pymc_model,
        replacements,
    )


def replace_variable_by_repeated_optimization_variable(
    pymc_model,
    name,
    xr_data: DataArray,
    n_repeat,
    extra_replacements: dict | None = None,
):
    """Replace a variable in the PyMC model graph with a repeated optimization variable.

    Parameters
    ----------
    pymc_model : Model
        The PyMC model containing the variable to replace.
    name : str
        The name of the variable to replace.
    xr_data : xarray.DataArray
        The xarray DataArray containing the data for the variable,
        used to determine the shape and dimensions of the optimization variable.
    n_repeat : int
        The number of times to repeat the optimization variable along the new dimension.
    extra_replacements : dict, optional
        Additional replacements to apply to the model graph, by default None.

    Returns
    -------
    tuple
        A tuple containing the repeated optimization variable (as an xtensor)
        and the PyTensor graph of the model with the variable replaced.

    Raises
    ------
    AssertionError
        If `xr_data` does not have exactly 2 dimensions or
        if the first dimension of `xr_data` does not have size 1,
        which are required for the broadcasting to work correctly.

    """
    assert len(xr_data.shape) == 2, "Expected xr_data to have 2 dimensions"
    assert (
        xr_data.shape[0] == 1
    ), "Expected the first dimension of xr_data to have size 1 for broadcasting"

    input_flat = xtensor(
        name=f"{name}_flat",
        shape=(xr_data.size,),
        dims=(f"{name}_flat",),
    )

    repeated_values = pt.repeat(
        input_flat.values[None, ...],  # pylint: disable=E1101, no-member
        repeats=n_repeat,
        axis=0,
    )

    repeated_xtensor = as_xtensor(
        repeated_values,
        dims=xr_data.dims,
        name=f"{name}_repeated",
    )

    replacements = {name: repeated_xtensor}
    if extra_replacements:
        replacements.update(extra_replacements)

    return input_flat, pm.do(
        pymc_model,
        replacements,
    )


def replace_variable_by_constant(pymc_model, name: str, xr_data: DataArray):
    """Replace a model variable with a fixed xarray-backed xtensor.

    Parameters
    ----------
    pymc_model : Model
        The PyMC model containing the variable to replace.
    name : str
        Name of the variable to replace.
    xr_data : xarray.DataArray
        Constant data used as replacement. Its shape and dims must be
        compatible with the target variable in the graph.

    Returns
    -------
    Model
        A copied PyMC model graph where ``name`` is replaced by ``xr_data``.
    """
    input_variable = as_xtensor(
        pt.as_tensor_variable(xr_data.values),
        dims=xr_data.dims,
        name=name,
    )

    return pm.do(
        pymc_model,
        {name: input_variable},
    )


def add_grad_to_graph(f, x) -> Variable:
    """Add the gradient of f with respect to x to the graph, returning the gradient variable.

    Parameters
    ----------
    f : Variable
        The variable representing the function for which to compute the gradient.
    x : Variable
        The variable with respect to which to compute the gradient.

    Returns
    -------
    Variable
        The variable representing the gradient of `f` with respect to `x`.
    """
    f_tensor = rewrite_graph(
        f.values, include=("lower_xtensor", "canonicalize", "stabilize")
    )
    f_grad = pt.grad(f_tensor, x)

    return f_grad


def function_with_grad(x: Variable, y: Variable) -> Function:
    """Compile a PyTensor function that returns both the value of `y` and
    its gradient with respect to `x`.

    Parameters
    ----------
    x : Variable
        The input variable with respect to which to compute the gradient.
    y : Variable
        The output variable for which to compute the value and gradient.

    Returns
    -------
    Function
        A compiled PyTensor function that takes `x` as input and
        returns a tuple of (`y`, gradient of `y` with respect to `x`).
    """
    y_grad = add_grad_to_graph(y, x)

    return compile_function(
        inputs=[x],
        outputs=[y, y_grad],
    )


def define_constraint_function(x, constraint_fun: Callable, constraint_type="eq"):
    """Define a constraint function for optimization, including its Jacobian.

    Parameters
    ----------
    constraint_fun : Variable
        The variable representing the constraint function.
    x : Variable
        The variable with respect to which to compute the Jacobian.
    constraint_type : str, optional
        The type of constraint ("eq" for equality, "ineq" for inequality), by default "eq".

    Returns
    -------
    dict
        A dictionary containing the compiled constraint function and
        its Jacobian, suitable for use in optimization routines
        that require callable functions for constraints.
    """

    constraint_tensor = constraint_fun(x)
    constraint_fun_jac = add_grad_to_graph(constraint_tensor, x)

    # Compile symbolic => python callables
    compiled_fun = compile_function(
        inputs=[x],
        outputs=constraint_tensor,
        # on_unused_input="ignore",
    )
    compiled_jac = compile_function(
        inputs=[x],
        outputs=constraint_fun_jac,
        # on_unused_input="ignore",
    )

    return {
        "type": constraint_type,
        "fun": compiled_fun,
        "jac": compiled_jac,
    }
