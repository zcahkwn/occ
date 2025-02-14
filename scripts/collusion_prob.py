import numpy as np
from math import comb

def collusion_2(N,shard_sizes: list[int]):
    n1, n2 = shard_sizes
    return comb(n1,n1+n2-N)/comb(N,n2)

def collusion_3(N:int,shard_sizes: list[int]):
    n1, n2, n3 = shard_sizes
    sum_=0
    for i in range(max(0,N-n1-n3),min(N-n1+1,n2+1)):
        sum_ += comb(N-n1,i)*comb(n1,n2-i)*comb(n1+i,n3+n1+i-N)/(comb(N,n2)*comb(N,n3))
    return sum_



def collusion_4(N:int, shard_sizes: list[int]):
    n1, n2, n3, n4 = shard_sizes
    sum_=0
    for i in range(max(0,N-n1-n3-n4),min(N-n1+1,n2+1)):
        for j in range(max(0,N-n1-n4-i),min(N-n1-i+1,n3+1)):
            sum_ += comb(N-n1,i)*comb(n1,n2-i)*comb(N-n1-i,j)*comb(n1+i,n3-j)*comb(n1+i+j,n4+n1+i+j-N)/(comb(N,n2)*comb(N,n3)*comb(N,n4))
    return sum_
# print("for m=4: ",collusion_4(10,[4,4,3,2]))



def collusion_5(N:int, shard_sizes: list[int]):
    n1, n2, n3, n4, n5 = shard_sizes
    sum_=0
    for i in range(max(0,N-n1-n3-n4-n5),min(N-n1+1,n2+1)):
        for j in range(max(0,N-n1-n4-n5-i),min(N-n1-i+1,n3+1)):
            for k in range(max(0,N-n1-n5-i-j),min(N-n1-i-j+1,n4+1)):
                sum_ += comb(N-n1,i)*comb(n1,n2-i)*comb(N-n1-i,j)*comb(n1+i,n3-j)*comb(N-n1-i-j,k)*comb(n1+i+j,n4-k)*comb(n1+i+j+k,n5+n1+i+j+k-N)/(comb(N,n2)*comb(N,n3)*comb(N,n4)*comb(N,n5))
    return sum_

# print("for m=5: ",collusion_5(10,[4,4,3,2,1]))
