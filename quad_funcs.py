#helper functions
import math
from sympy.ntheory import factorint
import numpy as np
from progressbar import *   
from tqdm import tqdm
import random

from operator import mul, add
from functools import reduce

import gmpy2
from gmpy2 import mpz

#from numba import jit


## matrix functions




#@jit(nopython=True)
def gcd(x, y): 
    if y > x: return gcd(y, x)
    while(y): 
        x, y = y, x % y 
    return x

#computes gcd when b is very large
#@jit(nopython=True)
def gcd_large(a, b) :
    
    g = gmpy2.gcd(mpz(a), mpz(b))
    
    return int(g)



#@jit(nopython=True)
def erat_sieve(B):
    # Use Sieve of Eratosthenes to find all primes <= B
    is_prime = [True for i in range(B+1)]
    
    for i in range(math.ceil(math.sqrt(B))):
        if i == 0 or i == 1:
            is_prime[i] = False
        
        if is_prime[i]:
            for j in range(i*2, B + 1, i): #goes through all other multiples of i
                is_prime[j] = False

    
    return [p for p in range(B+1) if is_prime[p]]

#@jit(nopython=True)
def add_cols(M, a, b):
    
    return np.array([(M[i][a] + M[i][b]) % 2 for i in range(len(M))])

#@jit(nopython=True)
def find_pivot(M, col):
    for i, row in enumerate(range(len(M))):
        if M[row][col] == 1: return i
    
    return None

#@jit(nopython=True)
def reduce_matrix(M):
    #taken from https://www.sciencedirect.com/science/article/pii/074373159190115P
    #Assume M is K+1 x K
    
    ##Gaussian elimination of matrix
    K = len(M[0])
    
    marked = [0] * (K+1)
    pivots = [None] * K
    
    

    for j in tqdm(range(K)):
    #for j in range(K):
        i = find_pivot(M, j)
        if i is not None:
            marked[i] = True
            pivots[j] = i
            for k in range(K):
                if M[i][k] == 1 and j != k:
                    #print("k=", k)
                    M[:, k] = add_cols(M, j, k)
    
    #finding dependent rows
    print("finding dependent rows")
    rows = []
    for i in tqdm(range(K+1)):
    #for i in range(K+1):
        if not marked[i]:
            indices = [i]
            for j in range(K):
                if M[i][j] == 1:
                    indices.append(pivots[j])
            rows.append(indices)
                    
    return M, rows[0]

#@jit(nopython=True)
def mod2(xs):
    return np.array([x % 2 for x in xs])

#@jit(nopython=True)
def find_a(a, p):
    """
    Find x such that x^2 = a (mod p)
    """
    t = 0
    while jacobi(t**2 - a, p) != -1:
        t = random.randint(0, p-1)
    
    p_2 = p**2
    
    i = (t**2 - a) % p_2
    j = (t + math.sqrt(i)) % p_2
    
    #x = j**(p+1)/2 % p**2
    """x = 1
    for i in range((p+1)//2):
        x  = (x * j) % p_2
    """
    
    x = bin_ladder(j, int((p+1)/2), p_2)
    
    return x 

#@jit(nopython=True)
def jacobi(a, m):
    """return jacobi(a/m)"""
    a = a % m 
    t = 1
    while a != 0:
        while a % 2 == 0:
            a /= 2
            if m % 8 in [3, 5]:
                t = -t
        
        a, m = m, a
        
        if a % 4 == 3 and m % 4 == 3:
            t = -t
        
        a = a % m
        
    if m == 1: return t
    
    return 0

#@jit(nopython=True)
def is_bsmooth(base_prod, y):
    # compare to trial division
    #based off https://math.stackexchange.com/questions/182398/smooth-numbers-algorithm
    
    k = base_prod
    
    
    g = gcd_large(y, k)
    #g = gcd(y, k)
        
    while g > 1:
        #solve for y = rg^e
        r = y
        while r % g == 0:
            r /= g
        
        if r == 1:
            return True
        
        y = r
        
        g = gcd_large(y, k)
        #g = gcd(y, k)
            
        
    
    return False

#def is_bmsooth2(based_prod, mod_tree, y):
    
    
#@jit(nopython=True)
def b_factor(base, y):
    factors = np.zeros(len(base))
    
    while y > 1:
        #x = max([b for b in base if b < math.sqrt(y) + 1])
        x = len(base)
        for i in range(len(base[:x])):
            
            if y % base[i] == 0:
                factors[i] += 1
                
                y = y/base[i]
                
                #break
                
                
    return factors

#@jit(nopython=True)
def b_factor2(base, base_prod, y):
    #attempt at a faster b_factoring method
    
    factors = np.zeros(len(base))
    
    
    while y > 1:
        g = gcd_large(base_prod, y)
        #print("\ng=", g)
        
        
        if g in base:
            factors[base.index(g)] += 1
        
            
        else:
            #print("subfactoring", g)
            sub_factors = b_factor(base, g)# doesnt work - call b_factor1??
            
            factors = list(map(add, factors, sub_factors))  #add elementwise
            
        
        y /= g
        
        #print("y=", y)
                                                            
        
    
    return np.array(factors)

#@jit(nopython=True)
def b_factor3(base, y):

    f = factorint(y)
    
    factors = [f.get(x, 0) for x in base]
    
    return factors
    
    
        
    
def bin_ladder(x, y, N):
    #compute x^y mod N
    
    y_string = bin(y)[2:]
    
    z = x
    
    for j in range(len(y_string)-1)[::-1]:

        z = z*z % N
        
        if y_string[j] == '1':
            z = (z*x) % N
    
    return z




#@jit(nopython=True)
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
    num_rows = len(active_M) #- 1
    num_cols = len(active_M[0]) - 1
    bpoint = 1
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
                    san_check = 0
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
                w = weight(active_M[a_row])
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
                    san_check = 0
                    for a_col in range(num_cols):
                        if active_M[a_row][a_col] == 1:
                            eliminated_cols[a_col] = 1
                            active_M[:, a_col] = [0]*num_rows
                            san_check+=1
                    assert(san_check == 1)
    
    #return modified matrix and keyset 
    num_elim_rows = weight(eliminated_rows)
    num_elim_cols = weight(eliminated_cols)
    
    """
    ret_M = np.array([[0]*(K-num_elim_cols)]*(K+1-num_elim_rows))
    
    
    i = 0
    j = 0
    i_off = 0
    j_off = 0
    for elim_row in eliminated_rows:
        if elim_row == 0:
            for elim_col in eliminated_cols:
                if elim_col == 0:
                    ret_M[i][j] =  M[i_off][j_off]
                    j+=1
                j_off+=1
            i+=1
        i_off+=1
        
    """
    
    ret_M = np.delete(M, [i for i in range(len(eliminated_rows)) if eliminated_rows[i]], 0) #delete rows
    ret_M = np.delete(ret_M, [i for i in range(len(eliminated_cols)) if eliminated_cols[i]], 0) #delete cols

    #return denser matrix, locations of valid rows and columns 
    return ret_M, [(elim_row==0) for elim_row in eliminated_rows]#, [(elim_col==0) for elim_col in eliminated_cols]

    #do normal reduction. If row i is dependent, return the ith true value in the array
              

def weight(A):
    return np.sum(A)
    #weight = 0
    #for i in A:
    #    weight += i
    #return weight
    
    
    
    

                
