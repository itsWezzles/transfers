import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

from P4_serial import *

# Slave's work
def slave(comm):
  # Get MPI data
  status = MPI.Status()

  # Always be ready to work
  while 1:
    # Receive row to work on from master
    row = comm.recv(source = 0, tag = MPI.ANY_TAG, status = status);
    # If master says is done via a tag of 1 ("done")
    if status.Get_tag():
      break
    # Send computed row back to master with row as tag
    comm.send([mandelbrot(col, minY + row * (maxY - minY) / height) for col in np.linspace(minX, maxX, width)], dest = 0, tag = row)

  return

# Master's (lack of) work
def master(comm):

  # Declare image variable
  image = np.zeros([height,width], dtype=np.uint16)

  # Get/declare MPI data
  size = comm.Get_size()
  status = MPI.Status()

  # First task is this point
  currentRow = 0

  # Defined as my "continue" working tag
  more = 0

  # First send all the processes a starting task
  for i in xrange(1, size):
    comm.send(currentRow, dest = i, tag = more)
    currentRow = currentRow + 1

  # Constantly listen for returned result by a slave while there is more work to give out
  while (currentRow < height):
    # Store received result into the row dictated by tag
    image[status.Get_tag(), :] = comm.recv(source = MPI.ANY_SOURCE, tag = MPI.ANY_TAG, status = status)

    # Get the rank of the slave who returned work and send back next job
    comm.send(currentRow, dest = status.Get_source(), tag = more)

    # Increment job counter
    currentRow = currentRow + 1

  # Defined as my "done" tag
  end = 1;

  # Send out the "done" notification
  for i in xrange(1, size):
    comm.send(None, dest = i, tag = end);

  return image


if __name__ == '__main__':

  # Get MPI data
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()

  # If master, start timer, start master parallel process and then print time/savec image when complete
  if rank == 0:
    start_time = MPI.Wtime()
    C = master(comm)
    end_time = MPI.Wtime()
    print "Time: %f secs" % (end_time - start_time)
    plt.imsave('Mandelbrot.png', C, cmap='spectral')
    plt.imshow(C, aspect='equal', cmap='spectral')
    plt.show()
  # If slave, start slave work
  else:
    slave(comm)
