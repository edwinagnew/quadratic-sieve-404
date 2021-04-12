import math
from sympy.ntheory import factorint
import numpy as np
from progressbar import *   
from tqdm import tqdm
import random

from operator import mul
from functools import reduce

import gmpy2
from gmpy2 import mpz

def matrix_structure_hueristic(M, target_dep=10, verbose=False):
    #as any dense matrix method, will require lots of memory due to dense encoding of inputs
        #likely mad python runtime do to python loop encoding 
    #method adapted from Pomerance and Smith 1992 on creating catastrophies
    #   at https://eudml.org/doc/231810
    #for K = pi(B) and M = F2^(K x K+1)
    #S is array of B-smooth numbers of length K
    #M is a matirix of exponents of B-smooth numbers 
    #assumed to have lower primes represented in the leftmost columns of M
    #   first access is identifying array of exponents of particular B-smooth num
    #       numbered 0:K identifying (first B-smooth):(final B-smooth);
    #   second access is identifying particular exponent of first access
    #       numbered 0:K-1 identifying (2):(greatest prime <= B)
    #target_dep is defined to be number of linear dependencies we want to discover
    K = len(M[0])
    #after element is used as pivot, consider row, col eliminated
    eliminated_rows = [0]*(K+1)
    eliminated_cols = [0]*(K)
    col_weights = [0]*(K)

    active_ind = math.ceil(0.05*K) 

    #step 0
    active_M = M[:][active_ind:]
    num_rows = len(active_M)
    num_cols = len(active_M[0])
    int bpoint = 1
    #continue work only on active columns
    #   deferring work on inactive columns

    while(bpoint):
        #step 1: eject columns of weight 0
        bpoint = 0
        for a_col in range(num_cols):
            #iterate through active columns 
            if eliminated_cols[active_ind + a_col] == 0:
                #skip eliminated cols
                if weight(active_M[:, a_col]) == 0:
                    eliminated_cols[active_ind + a_col] = 1
                    active_M[:, a_col] = [0]*num_rows
                    bpoint = 1
        #step 2: eject columns of weight 1 and corresponding row
        for a_col in range(num_cols):
            #iterate through active columns 
            if eliminated_cols[active_ind + a_col] == 0:
                #skip eliminated cols
                if weight(active_M[:, a_col]) == 1:
                    eliminated_cols[active_ind + a_col] = 1
                    int san_check = 0
                    for i in range(num_rows):
                        if active_M[i][a_col] == 1:
                            eliminated_rows[i] = 1
                            active_M[i] = [0]*num_cols
                            san_check+=1
                    assert(san_check==1)
                    active_M[:, a_col] = [0]*num_rows
                    bpoint = 1
        #step 3: eject excess (lin. dep.) rows
        row_surplus = (weight(eliminated_cols)-weight(eliminated_rows))
        heaviest_row_weight = -1
        heaviest_row = -1
        while(row_surplus > target_dep):
            for a_row in range(num_rows):
                int w = weight(active_M[a_row])
                if w > heaviest_row_weight:
                    heaviest_row_weight = w
                    heaviest_row = a_row
            if heaviest_row != -1:
                eliminated_rows[active_ind + heaviest_row] = 1
                active_M[heaviest_row] = [0]*num_cols
                bpoint = 1
        
            row_surplus = (weight(eliminated_cols)-weight(eliminated_rows))
        #and repeat steps 1-3 until no further reduction is possible
    
    #step 4: use weight 1 rows to eject columns and rows
    bpoint = 1
    while(bpoint):
        bpoint = 0
        for a_row in range(num_rows):
            if eliminated_rows[a_row] == 0:
                if weight(active_M[a_row]) == 1:
                    eliminated_rows[a_row] = 1
                    int san_check = 0
                    for a_col in range(num_cols):
                        if active_M[a_row][a_col] == 1:
                            eliminated_cols[a_col] = 1
                            active_M[:, a_col] = [0]*num_rows
                            san_check+=1
                    assert(san_check == 1)
    
    #return modified matrix and keyset 
    num_elim_rows = weight(eliminated_rows)
    num_elim_cols = weight(eliminated_cols)
    ret_M = [[0]*(K-num_elim_cols)]*(K+1-num_elim_rows)

    int i = 0
    int j = 0
    int i_off = 0
    int j_off = 0
    for elim_row in eliminated_rows:
        if elim_row == 0:
            for elim_col in eliminated_cols:
                if elim_col == 0:
                    ret_M[i][j] =  M[i_off][j_off]
                    j+=1
                j_off+=1
            i+=1
        i_off+=1

    #return denser matrix, locations of valid rows and columns 
    return ret_M, [(elim_row==0) for elim_row in eliminated_rows], [(elim_col==0) for elim_col in eliminated_cols]
              

def weight(A):
    int weight = 0
    for i in A:
        weight += i
    return weight
        
