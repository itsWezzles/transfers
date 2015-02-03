import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

from P4_serial import *

def parallel_susie(comm, p_root = 0):

  # get MPI variables for each processor
  rank = comm.Get_rank()
  size = comm.Get_size()

  # create storage for local computed image
  local_image = np.zeros([height,width], dtype=np.uint16)

  # compute Mandelbrot and store in respsective position of full image
  # Joe's np - > n(p+1) rows computation
  for i in xrange(0, height / size):
    # get row from rank, i
    row = i + rank * height /  size
    # use mandelbrot() to get row
    local_image[row,:] = [mandelbrot(col, minY + row * (maxY - minY) / height) for col in np.linspace(minX, maxX, width)]

  # reduce image in overlay method - to help clear confusion when doing Susie's 1 + Np  approach
  image = comm.reduce(local_image, root = p_root)
  return image


if __name__ == '__main__':
  # Get MPI data
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  # Set up timer, run parallel
  start_time = MPI.Wtime()
  image = parallel_susie(comm)
  end_time = MPI.Wtime()

  # If master, print and save image
  if rank == 0:
    print "Time: %f secs" % (end_time - start_time)
    plt.imsave('Mandelbrot.png', image, cmap='spectral')
    plt.imshow(image, aspect='equal', cmap='spectral')
    plt.show()
