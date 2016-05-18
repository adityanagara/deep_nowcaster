# Parallel benchmark.
#
# Example:
# time python parallel.py 3
#

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
import sys
import time

def make_heavy_op(myqueue, name):
  """Make an op that loops many times. This op reads off an int32 from "myqueue"
     and increments counter that many times. The return value is a Print op that
     final value of the counter."""

  looplimit = myqueue.dequeue()
  startval = tf.constant(0)
  condition = lambda i: tf.less(i, looplimit)
  increment = lambda i: tf.add(i, 1)
  result = control_flow_ops.While(condition, increment, [startval])
  resultprint = tf.Print(result, [name])
  return resultprint

def parallel_loops(parallelism, num_entries=10, duration=10**4):
  """Enqueues "num_entries*parallelism" heavy ops and tries to execute them
     in parallel, using "parallelism" threads"""
  q = tf.FIFOQueue(num_entries*parallelism, np.int32, shapes=())
  stuff = tf.placeholder(np.int32)
  addstuff = q.enqueue([stuff])
  sess = tf.InteractiveSession()
  for i in range(num_entries*parallelism):
    sess.run([addstuff], feed_dict={stuff:duration})

  sess.run([q.close()])
  ops = [make_heavy_op(q, "heavy op #"+str(i)) for i in range(parallelism)]

  runner = tf.train.QueueRunner(q, ops)
  
  start_time = time.time()
  threads = runner.create_threads(sess, start=True)
  for t in threads:
    t.join()

  elapsed_time = time.time() - start_time
  print 'done in %.2f, %.2f ops/sec' %(elapsed_time, (num_entries*parallelism)/elapsed_time)

if __name__=='__main__':
  parallel_loops(int(sys.argv[1]))

