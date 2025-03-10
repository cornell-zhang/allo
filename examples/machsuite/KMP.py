import allo
import os
import json
import pytest
import numpy as np
from allo.ir.types import int32, uint8, index
import allo.ir.types as T


def python_kmp(pattern, input):
    k = 0
    kmp_next = np.zeros(len(pattern), dtype=object)
    kmp_next[0] = 0

    for q in range(1, len(pattern)):

        while k > 0 and pattern[k] != pattern[q]: 
            k = kmp_next[k - 1]
 
        if pattern[q] == pattern[k]:
            k += 1
        kmp_next[q] = k
    
    matches = 0
    q = 0
    for i in range(len(input)):
        while (q > 0 and pattern[q] != input[i]):
            q = kmp_next[q - 1]
            
        if (pattern[q] == input[i]):
            q += 1
            
        if (q >= len(pattern)):
            matches += 1
            q = kmp_next[q - 1]

    return matches    


### allo implementation ###
def kmp(concrete_type, s, p):
    def kmp_kernal[

        T: (uint8, int32), S: uint8, P: uint8 
    ](pattern: "T[P]", input_str: "T[S]", kmp_next: "T[P]", matches: "T[1]"):
        
        k: index = 0
        x: index = 1
        
        for i in allo.grid((P - 1), name = "CPF"):
            while k > 0 and pattern[k] != pattern[x]:
                k = kmp_next[k-1]
                
            if pattern[k] == pattern[x]:
                k += 1
            kmp_next[x] = k
            x += 1
        
        q: index = 0
        for i in allo.grid(S, name = "KMP"):
            while (q > 0 and pattern[q] != input_str[i]):
                q = kmp_next[q-1]
              
            if (pattern[q] == input_str[i]):
                q += 1

            if q >= P:
                matches[0] += 1
                q = kmp_next[q-1]

            
            
    sch = allo.customize(kmp_kernal, instantiate = [concrete_type, s, p])

    return sch


def test_kmp():
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as f:
        psize = json.load(f)

    test_psize = "large"
    S = psize["kmp"][test_psize]["Str_len"]
    P = psize["kmp"][test_psize]["Patt_len"]

    concrete_type = uint8
    sch = kmp(concrete_type, S, P)

    #functional correctness checking
    Input_str  = np.random.randint(1, 5, size = S).astype(np.uint8)

    Pattern =  np.random.randint(1, 5, size = P).astype(np.uint8)


    print('hello')

    KMP_next = np.zeros(P).astype(np.uint8)
    debug_var = np.zeros(S).astype(np.uint8)


    kmp_matches = np.zeros(1).astype(np.uint8)
    mod = sch.build()

    kmp_matches_ref = python_kmp(Pattern, Input_str)

    mod(Pattern, Input_str, KMP_next, kmp_matches)
    np.testing.assert_allclose(kmp_matches[0], kmp_matches_ref, rtol=1e-5, atol=1e-5)


test_kmp()




