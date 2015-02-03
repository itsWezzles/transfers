from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import math
import time

from P3_serial import data_transformer

def parallel_tomo(data, comm, Transformer, n_phi, image_size, sample_size, p_root=0):

  # Get MPI Data
  rank = comm.Get_rank()
  size = comm.Get_size()

  # Scatter the arrays to all processes
  my_data = comm.scatter(data, root=p_root)

  # Declare storage for local image computation
  local_result = np.zeros((image_size,image_size), dtype=np.float64)

  # Reshape back to image dimensions
  my_data = my_data.reshape(n_phi / size, sample_size)

  # Calculate k with respect to the original image
  offset = rank * n_phi / size

  # For each row of scattered data
  for k in xrange(0, n_phi / size):
    # Compute the angle of this slice
    phi = -(k + offset) * math.pi / n_phi
    # Accumulate the back-projection
    local_result += Transformer.transform(my_data[k,:], phi)

  # Reduce the partial results to the master process, defaults to summation
  result = comm.reduce(local_result, root=p_root)
  return result

if __name__ == '__main__':
  # Metadata
  n_phi       = 2048   # The number of Tomographic projections
  sample_size = 6144   # The number of samples in each projection

  # Get MPI data
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  # Declare variables for all processors
  data, result = None, None
  # Read the projective data from file on master
  if rank == 0:
    data = np.fromfile(file='TomoData.bin', dtype=np.float64)
    data = data.reshape(size, n_phi * sample_size / size)

  # Allocate space for the tomographic image

  image_size = 512

  # If master, declare space for final result
  if rank == 0:
    result = np.zeros((image_size,image_size), dtype=np.float64)

  # Precompute a data_transformer
  Transformer = data_transformer(sample_size, image_size)

  # Start timer and run parallel code
  comm.barrier()
  p_start = MPI.Wtime()
  result = parallel_tomo(data, comm, Transformer, n_phi, image_size, sample_size)
  comm.barrier()
  p_stop = MPI.Wtime()

  # If master, print out time and save image
  if rank == 0:
    print "Parallel Time: %f secs" % (p_stop - p_start)
    plt.imsave('TomographicReconstruction.png', result, cmap='bone')
    raw_input("Any key to exit...")
