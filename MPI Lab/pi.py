from mpi4py import MPI
import numpy as np
import time

from piserial import mc_pi

def parallel_mc_pi(n, comm, p_root=0):
  '''Distributes the computation of @param n test points to compute pi to various processors
     Calls mc_pi which will be run per process'''
  size = comm.Get_size()

  # compute local answer
  myCount = mc_pi(n/size)
  print "rank: %d, myCount; %d)" % (rank, myCount)
  # Reduce the partial results to the root process
  totalCount = comm.reduce(myCount, op=MPI.SUM, root=p_root)
  if rank == 0:
    print "rank 0 total count" + str(totalCount)
  return totalCount


if __name__ == '__main__':
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  
  # use numPoints points in MC 
  numPoints = 200000
  comm.barrier()
  p_start = MPI.Wtime()
  p_answer = (4.0 * parallel_mc_pi(numPoints, comm)) / numPoints
  comm.barrier()
  p_stop = MPI.Wtime()

  # Compare to serial results on process 0
  if rank == 0:
    s_start = time.time()
    s_answer = (4 * mc_pi(numPoints)) / numPoints
    s_stop = time.time()
    print "Serial Time: %f secs" % (s_stop - s_start)
    print "Parallel Time: %f secs" % (p_stop - p_start)

    print "Serial Result   = %f" % s_answer
    print "Parallel Result = %f" % p_answer
    print "NumPy  = %f" % np.pi