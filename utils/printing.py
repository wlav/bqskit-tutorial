from bqskit.ir import Circuit

def print_circuit(circuit: Circuit) -> None:
    for op in circuit:
        if op.num_params == 0:
            print(op)
        else:
            print(op, [float(p) for p in op.params])