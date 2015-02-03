from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import math
import time

from P3_serial import data_transformer

def parallel_tomo(data, comm, Transformer, n_phi, image_size, sample_size, p_root=0):

  # Get MPI data
  rank = comm.Get_rank()
  size = comm.Get_size()

  # If master, send submatrix chunk to each
  if rank == p_root:
    for i in xrange(1, size):
      comm.send(data[i * n_phi/size:(i+1)*n_phi/size,:], dest=i)
  # Receive submatrix chunk if not master
  else:
      data = comm.recv(source=p_root)

  # Declare local space for image
  local_result = np.zeros((image_size,image_size), dtype=np.float64)

  # calculate k with respect to the original image
  offset = rank * n_phi / size

  # For each row of the data
  for k in xrange(0, n_phi / size):
    # Compute the angle of this slice
    phi = -(k + offset) * math.pi / n_phi
    # Accumulate the back-projection
    local_result += Transformer.transform(data[k,:], phi)

  # If not master, send result to root
  if rank != p_root:
    comm.send(local_result, dest=p_root)
  # If master, receive data of image and add (overlay) to current
  else:
    result = local_result
    for i in xrange(1, size):
      result += comm.recv(source=i)
    return result

if __name__ == '__main__':

  # Metadata
  n_phi       = 2048   # The number of Tomographic projections
  sample_size = 6144   # The number of samples in each projection

  # Get MPI data
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  # Declare variables for all
  data, result = None, None

  # Read in data on master
  if rank == 0:
    data = np.fromfile(file='TomoData.bin', dtype=np.float64)
    data = data.reshape(n_phi, sample_size)

  # Allocate space for the tomographic image
  image_size = 512

  # Allocate space for final result if master
  if rank == 0:
    result = np.zeros((image_size,image_size), dtype=np.float64)

  # Precompute a data_transformer
  Transformer = data_transformer(sample_size, image_size)

  # Calculate time and run parallel code
  comm.barrier()
  p_start = MPI.Wtime()
  result = parallel_tomo(data, comm, Transformer, n_phi, image_size, sample_size)
  comm.barrier()
  p_stop = MPI.Wtime()

  # If master, display time and save image
  if rank == 0:
    print "Parallel Time: %f secs" % (p_stop - p_start)
    plt.imsave('TomographicReconstruction.png', result, cmap='bone')
    raw_input("Any key to exit...")
