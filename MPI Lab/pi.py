from mpi4py import MPI
import numpy as np
import time

from piserial import mc_pi

def parallel_mc_pi(n, comm, p_root=0):
  '''Distributes the computation of @param n test points to compute pi to various processors
     Calls mc_pi which will be run per process'''
  rank = comm.Get_rank()
  size = comm.Get_size()

  # compute local answer
  myCount = mc_pi(n/size, seed=rank)

  # sanity check print statements
  # print "rank: %d, myCount: %d" % (rank, myCount)

  # Reduce the partial results to the root process
  totalCount = comm.reduce(myCount, op=MPI.SUM, root=p_root)
  return totalCount


if __name__ == '__main__':
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  
  # use numPoints points in MC 
  numPoints = 10000000
  comm.barrier()
  p_start = MPI.Wtime()

  # all processors return, some wil return None
  # could replace this with allReduce but unnecessary
  p_answer = parallel_mc_pi(numPoints, comm)
  comm.barrier()
  p_stop = MPI.Wtime()

  # we only care a out rank 0 root answer
  if rank == 0:
    p_answer = (4.0 * p_answer) / numPoints

  # Compare to serial results on process 0
  if rank == 0:
    s_start = time.time()
    s_answer = (4.0 * mc_pi(numPoints)) / numPoints
    s_stop = time.time()
    print "Serial Time: %f secs" % (s_stop - s_start)
    print "Parallel Time: %f secs" % (p_stop - p_start)

    print "Serial Result   = %f" % s_answer
    print "Parallel Result = %f" % p_answer
    print "NumPy  = %f" % np.pi