import numpy as np
import time

def mc_pi(n):
  '''Computes pi using an Monte Carlo method by testing
     @param n random points inside unit square for being inside unit circle'''
  
  # Set the random seed
  np.random.seed(0) 

  # Set counter
  count = 0

  # run MC
  for i in xrange(n):
    testPt = np.random.uniform(-1, 1, size=2)
    if np.linalg.norm(testPt) < 1:
      count += 1

  return count

if __name__ == '__main__':

  # use numPoints points in MC 
  numPoints = 200000.

  s_start = time.time()
  s_answer = 4.0 * mc_pi(numPoints) / numPoints
  s_stop = time.time()
  print "Serial Time: %f secs" % (s_stop - s_start)

  print "Serial Result   = %f" % s_answer
