"""
The non-elegant way to calculate the collusion probability for m shards
"""

from math import comb
import numpy as np

def collusion_2(N,shard_sizes: list[int]):
    n1, n2 = shard_sizes
    if n1 + n2 < N:
        return np.nan
    return comb(n1,n1+n2-N)/comb(N,n2)

def collusion_3(N:int,shard_sizes: list[int]):
    n1, n2, n3 = shard_sizes
    sum_=0
    for i in range(max(0,N-n1-n3),min(N-n1+1,n2+1)):
        sum_ += comb(N-n1,i)*comb(n1,n2-i)*comb(n1+i,n3+n1+i-N)/(comb(N,n2)*comb(N,n3))
    return sum_
# print("for m=3, N=10, [4,3,3], using method 1: ",collusion_3(10,[4,3,3]))

def collusion_3_method_2(N:int,n1,n2,n3):
    sum_=0
    for i in range(max(N-n1,n3,n2),min(n2+n3,N)+1):
        sum_ += comb(n1,n1+i-N)*comb(i,n2)*comb(n2,n2+n3-i)/(comb(N,n2)*comb(N,n3))
    return sum_
# print("using method 2: ",collusion_3_method_2(10,2,6,3))

def collusion_3_method_3(N:int,shard_sizes: list[int]):
    sum_=0
    n1,n2,n3 = shard_sizes
    def f2(k,n1,n2):
        return comb(n1,n1+n2-k)/comb(k,n2)
    for i in range(max(N-n3,n1,n2),min(n1+n2,N)+1):
        sum_ += comb(n3,n3+i-N)*comb(i,n2)*comb(i,n1)*f2(i,n1,n2)/(comb(N,n2)*comb(N,n1))
    return sum_
print("using method 3: ",collusion_3_method_3(10,[2,6,5]))

def collusion_4(N:int, shard_sizes: list[int]):
    n1, n2, n3, n4 = shard_sizes
    sum_=0
    for i in range(max(0,N-n1-n3-n4),min(N-n1+1,n2+1)):
        for j in range(max(0,N-n1-n4-i),min(N-n1-i+1,n3+1)):
            sum_ += comb(N-n1,i)*comb(n1,n2-i)*comb(N-n1-i,j)*comb(n1+i,n3-j)*comb(n1+i+j,n4+n1+i+j-N)/(comb(N,n2)*comb(N,n3)*comb(N,n4))
    return sum_
# print("for m=4: ",collusion_4(10,[4,4,3,2]))

def collusion_4_method_2(N:int,shard_sizes: list[int]):
    n1, n2, n3, n4 = shard_sizes
    sum_=0
    for i in range(max(N-n1,n2,n3,n4),min(n2+n3+n4,N)+1):
        for j in range(max(N-n1-i,n3,n4,i-n2),min(n3+n4,i)+1):
            sum_ += comb(n1,n1+i-N)*comb(n2,n2+j-i)*comb(i,n2)*comb(j,n3)*comb(n3,n3+n4-j)/(comb(N,n2)*comb(N,n3)*comb(N,n4))
    return sum_
# print("for m=4: ",collusion_4_method_2(10,[7,2,1,1]))

def collusion_4_method_3(N:int,shard_sizes: list[int]):
    n1, n2, n3, n4 = shard_sizes
    sum_=0
    def f2(k,n1,n2):
        return comb(n1,n1+n2-k)/comb(k,n2)
    def f3(k,n1,n2,n3):
        sum_3=0
        for j in range(max(k-n3,n1,n2),min(n1+n2,k)+1):
            sum_3 += comb(n3,n3+j-k)*comb(j,n2)*comb(j,n1)*f2(j,n1,n2)/(comb(k,n2)*comb(k,n1))
        return sum_3
    for i in range(max(N-n4,n1,n2,n3),min(n1+n2+n3,N)+1):
        sum_ += comb(i,n3)*comb(n4,n4+i-N)*comb(i,n1)*comb(i,n2)*f3(i,n1,n2,n3)/(comb(N,n1)*comb(N,n2)*comb(N,n3))
        return sum_
# print("using method 3: ",collusion_4_method_3(10,[2,6,5,3]))

def collusion_5(N:int, shard_sizes: list[int]):
    n1, n2, n3, n4, n5 = shard_sizes
    sum_=0
    for i in range(max(0,N-n1-n3-n4-n5),min(N-n1+1,n2+1)):
        for j in range(max(0,N-n1-n4-n5-i),min(N-n1-i+1,n3+1)):
            for k in range(max(0,N-n1-n5-i-j),min(N-n1-i-j+1,n4+1)):
                sum_ += comb(N-n1,i)*comb(n1,n2-i)*comb(N-n1-i,j)*comb(n1+i,n3-j)*comb(N-n1-i-j,k)*comb(n1+i+j,n4-k)*comb(n1+i+j+k,n5+n1+i+j+k-N)/(comb(N,n2)*comb(N,n3)*comb(N,n4)*comb(N,n5))
    return sum_
# print("for m=5: ",collusion_5(10,[4,4,3,2,1]))

def collusion_5_method_2(N:int,shard_sizes: list[int]):
    n1, n2, n3, n4, n5 = shard_sizes
    sum_=0
    for i in range(max(N-n1,n2,n3,n4,n5),min(n2+n3+n4+n5,N)+1):
        for j in range(max(N-n1-i,n3,n4,n5,i-n2),min(n3+n4+n5,i)+1):
            for k in range(max(N-n1-i-j,n4,n5,j-n3),min(n4+n5,j)+1):
                sum_ += comb(n1,n1+i-N)*comb(n2,n2+j-i)*comb(n3,n3+k-j)*comb(n4,n4+n5-k)*comb(i,n2)*comb(j,n3)*comb(k,n4)/(comb(N,n2)*comb(N,n3)*comb(N,n4)*comb(N,n5))
    return sum_
# print("for m=5: ",collusion_5_method_2(10,[4,4,1,1,1]))
                        
def collusion_m(N: int, shard_sizes: list[int]) -> float:
    m = len(shard_sizes)
    if m < 2:
        raise ValueError("Need at least 2 shards")
    
    denom = 1
    for i in range(1, m):
        denom *= comb(N, shard_sizes[i])
    
    def rec(level: int, prev: int, sum_k: int, prod: int) -> int:
        if level > m - 2: # last level
            # multiply by binom{n_{m-1}} {n_{m-1} + n_m - k_{m-2}}
            return prod * comb(shard_sizes[m-2], shard_sizes[m-2] + shard_sizes[m-1] - prev) 
            
        q = N - shard_sizes[0] - sum_k # q_i = N - n1 - (k_1 + ... + k_{i-1})
        p = sum(shard_sizes[level:]) # p_i = sum_{j=i+1}^m n_j
        
        # lower bound
        if level == 1:
            # first summation index, no r_i appears.
            lower = max(shard_sizes[m-1], q)
        else:
            # For level i (>=2)
            r = prev - shard_sizes[level - 1] # r_i = k_{i-1} - n_i
            lower = max(shard_sizes[m-1], q, r)
        
        # Upper bound 
        upper = min(p, prev)
        s = 0
        for k in range(lower, upper + 1):
            if level == 1:
                # comb(n1, n1 + k - N) * comb(k, n2)
                current_factor = comb(shard_sizes[0], shard_sizes[0] + k - N) * comb(k, shard_sizes[1])
            else:
                # For level i (>=2):
                # comb(n_i, n_i + k - k_{i-1}) * comb(k, n_{i+1})
                current_factor = comb(shard_sizes[level - 1], shard_sizes[level - 1] + k - prev) * comb(k, shard_sizes[level])
            s += rec(level + 1, k, sum_k + k, prod * current_factor)
        return s

    total = rec(1, N, 0, 1)
    return total / denom
# print("for m=4: ",collusion_m(10,[3,4,3,5]))
