"""One-way sensitivity analysis (tornado chart data).

Sensitivity analysis examines how the uncertainty in model inputs
drives the uncertainty in model outputs.  The simplest and most widely
used variant is **one-way (one-at-a-time, OAT) sensitivity**: each
input is varied over a symmetric range around its base value while all
other inputs are held at their base values, and the change in output
is recorded.

Plotting the output ranges as horizontal bars sorted by magnitude
yields the familiar **tornado diagram**, a staple of cost/risk models
(e.g. @RISK, Crystal Ball outputs, NASA cost-risk guidance).

Example
-------
>>> from risk_analysis.quantitative import one_way_sensitivity, tornado_data
>>> def model(inputs):
...     return inputs["revenue"] - inputs["cost"]
>>> result = one_way_sensitivity(
...     base_inputs={"revenue": 1000.0, "cost": 600.0},
...     model=model,
...     vary_pct=0.2,
... )
>>> result["revenue"]["low"], result["revenue"]["high"]
(200.0, 600.0)
>>> tornado_data(result)[0][0]
'revenue'
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple


def one_way_sensitivity(
    base_inputs: Dict[str, float],
    model: Callable[[Dict[str, float]], float],
    vary_pct: float = 0.1,
    steps: int = 2,
) -> Dict[str, Dict[str, float]]:
    """Run one-at-a-time sensitivity analysis on a scalar model.

    Parameters
    ----------
    base_inputs:
        Mapping from input name to base value.  Values must be
        numeric.
    model:
        Callable accepting the input dictionary and returning a scalar
        model output (cost, NPV, loss, ...).
    vary_pct:
        Relative variation applied to each input, e.g. ``0.2`` varies
        every input in ``[-20%, +20%]`` around its base value.
    steps:
        Number of sample points on each side of the base value
        (``steps >= 1``).  The reported ``low``/``high`` are the
        extremes of the swept range.

    Returns
    -------
    dict
        Mapping from input name to a dictionary with keys:

        * ``base_output`` -- model output with all inputs at base.
        * ``low`` -- minimum output over the swept range.
        * ``high`` -- maximum output over the swept range.
        * ``low_input`` / ``high_input`` -- input values producing the
          extreme outputs.
        * ``swing`` -- ``high - low`` (absolute output range).

    Raises
    ------
    ValueError
        If ``vary_pct <= 0`` or ``steps < 1`` or the input dictionary
        is empty.

    Examples
    --------
    See module docstring.
    """
    if vary_pct <= 0:
        raise ValueError("vary_pct must be positive")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if not base_inputs:
        raise ValueError("base_inputs must not be empty")

    base_output = float(model(dict(base_inputs)))
    results: Dict[str, Dict[str, float]] = {}

    for name, base_value in base_inputs.items():
        sweep = [
            base_value * (1.0 + vary_pct * i / steps)
            for i in range(-steps, steps + 1)
        ]
        outputs = []
        for value in sweep:
            trial = dict(base_inputs)
            trial[name] = value
            outputs.append(float(model(trial)))
        low_output = min(outputs)
        high_output = max(outputs)
        results[name] = {
            "base_output": base_output,
            "low": low_output,
            "high": high_output,
            "low_input": sweep[outputs.index(low_output)],
            "high_input": sweep[outputs.index(high_output)],
            "swing": high_output - low_output,
        }
    return results


def tornado_data(
    sensitivity: Dict[str, Dict[str, float]],
) -> List[Tuple[str, float, float]]:
    """Convert sensitivity results into sorted tornado-chart data.

    Parameters
    ----------
    sensitivity:
        Output of :func:`one_way_sensitivity`.

    Returns
    -------
    list of tuple
        ``(input_name, low, high)`` tuples sorted by descending output
        swing -- ready to be plotted as a tornado diagram (largest bar
        at the top).

    Examples
    --------
    >>> data = tornado_data({
    ...     "a": {"low": 0, "high": 10, "swing": 10,
    ...           "base_output": 5, "low_input": 0, "high_input": 1},
    ...     "b": {"low": 2, "high": 4, "swing": 2,
    ...           "base_output": 5, "low_input": 0, "high_input": 1},
    ... })
    >>> data[0][0]
    'a'
    """
    rows = [
        (name, values["low"], values["high"])
        for name, values in sensitivity.items()
    ]
    def _swing(row):
        return row[2] - row[1]
    rows.sort(key=_swing, reverse=True)
    return rows
