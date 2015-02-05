from mpi4py import MPI
import numpy as np
import time

def get_big_arrays():
  '''Generate two big random arrays.''' 
  N = 10    # A big number, the size of the arrays.
  np.random.seed(0)  # Set the random seed
  return np.random.random(N), np.random.random(N)

if __name__ == '__main__':
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  # Get big arrays on process 0
  a, b = None, None
  if rank == 0:
    a, b = get_big_arrays()

  a = comm.scatter(a)
  b = comm.scatter(b)
  temp = comm.scatter(a)

  print "Rank: " + str(rank) + " A: " + str(a) + " B: " + str(b) + " temp: " + str(temp)

  
