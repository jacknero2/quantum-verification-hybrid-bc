import itertools
from sympy import conjugate
import numpy as np
import sympy as sym


'''
Given the number of the qubits, generate the symbols
'''
def generate_symbols(n : int) -> list[sym.Symbol]:
    return [sym.Symbol('z' + str(i), complex=True) for i in range(2**n)]


'''
Generate variables
'''
def generate_variables(Z: list[sym.Symbol]) -> list[sym.Symbol]:
    variables = Z + [z.conjugate() for z in Z]
    return variables


'''
Given the variables we generate all possible terms for the polynomial
'''
def generate_terms(variables : list[sym.Symbol], deg=2) -> list[tuple]:
    p = []
    for d in range(deg + 1):
        p += itertools.combinations_with_replacement(variables, d) # get all possible combinations of variables
    return p


def construct_complex_coefficients(Re_a, Im_a):

    Re_a = np.array(Re_a)
    Im_a = np.array(Im_a)

    if Re_a.shape != Im_a.shape:
        raise ValueError("The real and imaginary parts must have the same shape.")

    a = Re_a + 1j * Im_a
    return a


'''
Function to check if a term contains both zi and conjugate(zi)
'''
def is_real_term(term):
    from collections import defaultdict
    counts = defaultdict(int)
    for elem in term:
        # Usa le stringhe per gestire i simboli e i loro coniugati
        elem_str = str(elem)
        conjugate_str = str(conjugate(elem))

        if conjugate_str in counts:
            counts[conjugate_str] -= 1
        else:
            counts[elem_str] += 1
    # Un termine è reale se tutti i conteggi sono zero
    return all(count == 0 for count in counts.values())

'''
Given the list of terms, we sort them
'''
def sort_terms(terms):
    sorted_terms = sorted(
        terms,
        key=lambda term: (
            len(term) != 0,  # La tupla vuota al primo posto
            -len(term),  # Termini di grado più alti prima
            (0 if len(term) % 2 == 0 and is_real_term(term) else 1),  # Termini reali di grado pari prima
            [str(e) for e in term]  # Ordine alfabetico per default
        )
    )
    return sorted_terms

def apply_op(sample_state, op):
    return np.dot(sample_state, op)


'''
Given the sampled term's values, separate into real and immaginary parts
'''
def separate_real_imag(sampled_terms) -> list[list[tuple]]:
    values = [[(value.real,value.imag) for value in sample] for sample in sampled_terms]
    return np.array(values)




def generate_barrier_polynomial(coeffs: list, terms:list, variables: list[sym.Symbol]) -> sym.Poly:
    p = [np.prod([t for t in term]) for term in terms]
    p = np.sum([coeffs[i] * p[i] for i in range(len(coeffs))])
    return sym.poly(p, variables, domain=sym.CC)

