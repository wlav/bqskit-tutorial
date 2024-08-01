from bqskit import Circuit
from bqskit.compiler import Compiler
from bqskit.ir.gates import PauliZGate
from bqskit.passes.synthesis import WalshDiagonalSynthesisPass as Decomp
from bqskit.passes import ScanningGateRemovalPass as Scan


def replace_pauliz(circuit: Circuit) -> Circuit:
    rebuilt = Circuit(circuit.num_qudits)
    for op in circuit:
        if isinstance(op.gate, PauliZGate):
            diagonal = Circuit.from_operation(op)
            with Compiler() as compiler:
                diagonal = compiler.compile(diagonal, [Decomp(), Scan()])
            rebuilt.append_circuit(diagonal, op.location)
        else:
            rebuilt.append_gate(op.gate, op.location)
    return rebuilt