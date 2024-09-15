"""This module implements the DiagonalCost and DiagonalCostGenerator."""
from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Sequence

from numpy import abs
from numpy import vdot
from numpy import float64
from numpy._typing import NDArray

from bqskit.ir.opt.cost.differentiable import DifferentiableCostFunction
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem
from bqskit.qis.unitary.unitary import RealVector

from bqskit.utils.math import diagonal_distance

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit
    from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
    from bqskit.ir.opt.cost.function import CostFunction


class DiagonalCost(DifferentiableCostFunction):
    """
    The DiagonalCost CostFunction implementation.

    The DiagonalCostFuction is a map from circuit to a cost value that is
    based on the Hilbert-Schmidt inner product. Concretely, it is the 
    distance from being invertable by a purely diagonal matrix. This 
    function is global-phase-aware, meaning that the cost is zero if the
    target and circuit unitary differ only by a global phase.
    """
    def __init__(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector | StateSystem,
    ) -> None:
        """
        Construct a DiagonalCost.

        Args:
            circuit (Circuit): The circuit to evaluate.

            target (UnitaryMatrix | StateVector | StateSystem): The target
                unitary to compare against.
        """
        self.circuit = circuit
        self.target = target

    def get_cost(self, params: list[float] = []) -> float:
        """Return the cost given the input parameters."""
        remainder = self.circuit.get_unitary(params) @ self.target.conj().T
        return diagonal_distance(remainder)
    
    def _state_infidelity_jac(self, u, m, j):
        d = vdot(u, m)  # Equivalent of state_dot in Rust
        infidelity = 1.0 - abs(d)**2
        d_infidelity = [
            -2.0 * (d.real * vdot(u, dv).real + d.imag * vdot(u, dv).imag)
            for dv in j
        ]
        return infidelity, d_infidelity
    
    def get_grad(self, params: RealVector) -> NDArray[float64]:
        m, j = self.circuit.get_unitary_and_grad(params)
        _, grad = self._state_infidelity_jac(self.target, m, j)
        return grad


class DiagonalCostGenerator(CostFunctionGenerator):
    """
    The DiagonalCostGenerator class.

    This generator produces configured DiagonalCost functions.
    """
    def gen_cost(
        self,
        circuit: Circuit,
        target: UnitaryMatrix | StateVector | StateSystem,
    ) -> CostFunction:
        """Generate a CostFunction, see CostFunctionGenerator for more info."""
        return DiagonalCost(circuit, target)
