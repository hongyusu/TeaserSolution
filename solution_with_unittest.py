
#!/usr/bin/python


'''
Python script to solve the teaser puzzle from Zalando
document for this script can be generate with $pydoc solution
unit text for this script can be triggled with $python solutin.py -v
'''

__author__ = 'Hongyu Su'
__version__ = '1.0'
__else__ = 'other information can be documented here'

import unittest
import doctest
import math
import numpy as np
import matplotlib.pylab as plt
import scipy.interpolate
from scipy.stats import norm
from scipy.stats import lognorm
from nose.tools import *

# data as global variable
riverGPS     =[
  (52.529198,13.274099),
  (52.531835,13.29234),
  (52.522116,13.298541),
  (52.520569,13.317349),
  (52.524877,13.322434),
  (52.522788,13.329),
  (52.517056,13.332075),
  (52.522514,13.340743),
  (52.517239,13.356665),
  (52.523063,13.372158),
  (52.519198,13.379453),
  (52.522462,13.392328),
  (52.520921,13.399703),
  (52.515333,13.406054),
  (52.514863,13.416354),
  (52.506034,13.435923),
  (52.496473,13.461587),
  (52.487641,13.483216),
  (52.488739,13.491456),
  (52.464011,13.503386)]

satelliteGPS = [
  (52.590117,13.39915),
  (52.437385,13.553989)] 

gateGPS      = (52.516288,13.377689)
startGPS     = (52.434011,13.274099)
stopGPS      = (52.564011,13.554099)



def dist(x1,y1, x2,y2, x3,y3):
  '''
  compute distance from a point to line segment
  x3,y3 is the point
  perform the following test
  >>> dist(-7.83434151195,17.378188238,-17.5348765366,0.375603802,-0.0,0.0)
  15.416540040627943
  '''
  px = x2-x1
  py = y2-y1
  something = px*px + py*py
  u =  ((x3 - x1) * px + (y3 - y1) * py) / float(something)
  if u > 1:
    u = 1
  elif u < 0:
    u = 0
  x = x1 + u * px
  y = y1 + u * py
  dx = x - x3
  dy = y - y3
  dist = math.sqrt(dx*dx + dy*dy)
  return dist

def prob_river(pointPOS,riverPOS):
  '''
  compute probability according to Gaussian distribution base on river
  '''
  mu    = 0
  delta = 2.730/1.96
  min_d = 10e10
  for i in range(1,len(riverPOS)):
    d = dist(riverPOS[i-1][0],riverPOS[i-1][1],riverPOS[i][0],riverPOS[i][1],pointPOS[0],pointPOS[1])
    if min_d > d: min_d =d
  return min_d,norm.pdf(min_d,mu,delta)

def prob_gate(pointPOS,gatePOS):
  '''
  compute probability according to lognormal distribution base on gate
  '''
  d       = math.sqrt((gatePOS[0]-pointPOS[0])**2+(gatePOS[1]-pointPOS[1])**2)
  mu      = (2*math.log(4.700) + math.log(3.877)) / float(3)
  delta   = math.sqrt(2/3*(math.log(4.7)-math.log(3.877)))
  return d,lognorm.pdf(d,mu,delta)

def prob_satellite(pointPOS,satellitePOS):
  '''
  compute probability according to Gaussian distribution for satellite
  '''
  d = dist(satellitePOS[0][0],satellitePOS[0][1],satellitePOS[1][0],satellitePOS[1][1],pointPOS[0],pointPOS[1])
  mu    = 0
  delta = 2.400/1.96
  return d,norm.pdf(d,mu,delta)

def GPS2POS((lat,lng)):
  '''
  transform from GPS to coordinate system (POS)
  perform the following test:
  >>> GPS2POS((52.516288,13.377689))
  (-6.48982764810209, 9.159322471000536)
  '''
  return ((lng-startGPS[1]) * math.cos(startGPS[0]) * 111.323, (lat-startGPS[0]) * 111.323)

def POS2GPS((x,y)):
  '''
  transform from coordinate system (POS) to GPS
  '''
  return (y/111.323+startGPS[0], x/111.323/math.cos(startGPS[0]) + startGPS[1])

def transformation(riverGPS,satelliteGPS,gateGPS,startGPS,stopGPS):
  '''
  wrapper function to transform the distance from GPS to locations POS
  '''
  gatePOS  = GPS2POS(gateGPS)
  startPOS = GPS2POS(startGPS)
  stopPOS  = GPS2POS(stopGPS)
  satellitePOS = [ GPS2POS(point) for point in satelliteGPS ]
  riverPOS = [ GPS2POS(point) for point in riverGPS ]
  return riverPOS,satellitePOS,gatePOS,startPOS,stopPOS

def plot_res(res):
  ind = 0
  for i in range(2,10):
    ind += 1
    x,y,z = np.transpose(res[:,[0,1,i]])
    x=-x
    # Set up a regular grid of interpolation points
    xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    # Interpolate
    rbf = scipy.interpolate.Rbf(x, y, z, function='linear')
    zi = rbf(xi, yi)
    # plot
    subplot = plt.subplot(4, 2, ind)
    if i%2 ==0:
      subplot.set_title("Distance")
    else:
      subplot.set_title("Probability")
    subplot.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower',extent=[x.min(), x.max(), y.min(), y.max()])
    subplot.scatter(x, y, c=z)
    subplot.set_xlabel('X')
    subplot.set_ylabel('Y')
  plt.show()
  pass

def compute_joint_probability(ss,gatePOS,satellitePOS,riverPOS):
  '''
  compute joint probability of all point in the search space
  '''
  res = []
  for pointPOS in ss:
    gateD,gateP            = prob_gate(pointPOS,gatePOS)
    satelliteD,satelliteP  = prob_satellite(pointPOS,satellitePOS)
    riverD,riverP          = prob_river(pointPOS,riverPOS)
    try:
      res.append([pointPOS[0],pointPOS[1],gateD,gateP,satelliteD,satelliteP,riverD,riverP,gateD+satelliteD+riverD, gateP*satelliteP*riverP])
    except Exception as error:
      print error
  return np.array(res)

def show_result(res):
  '''
  show results on google map
  '''
  res    = res[np.lexsort((res[:,-1],))] # sort point by probability
  s      = ''
  for line in open('head') : s+=line
  for i in range(1,6):
    pointGPS = POS2GPS((res[-i,0],res[-i,1]))
    print i,pointGPS,res[-i,[3,5,7,9]].tolist()
    s += '[ %.6f,%.6f,%d],\n' % (pointGPS[0],pointGPS[1],i)
  for line in open('tail') : s+=line
  open('map.html','w').write(s)

def save_probability(res):
  '''
  save probability to file
  '''
  for i in range(res.shape[0]):
    point = POS2GPS((res[i,0],res[i,1]))
    res[i,0],res[i,1] = point[0],point[1]
  np.savetxt('gpsdata.csv',res)

def find_her():
  '''
  the function is designed to output locations and probabilities
  '''

  # transformation from GPS to relative distance
  riverPOS,satellitePOS,gatePOS,startPOS,stopPOS = transformation(riverGPS,satelliteGPS,gateGPS,startGPS,stopGPS)
  
  # define a search space of points, with scale KM as interval
  scale = 0.05
  scale = 0.5
  ss    = [(x,y) for x in np.arange(startPOS[0],stopPOS[0],-scale) for y in np.arange(startPOS[1],stopPOS[1],scale)]
  print "Number of sample points:\t", len(ss)

  # compute statistics: x,y,dist_gate,prob_gate,dist_satellite,prob_satellite,dist_river,prob_river,sum of distance,joint probability
  res = compute_joint_probability(ss,gatePOS,satellitePOS,riverPOS)

  # save probablity data to file
  save_probability(res)

  # plot
  #plot_res(res)

  # output
  show_result(res)

  pass


#--------------------- test with unittest ---------------------------
class TestMethods(unittest.TestCase):
  '''
  class to define a set of unit test with Python unittest module
  '''
  def test_dist(self):
    '''
    test the function dist()
    '''
    self.assertEqual(dist(-7.83434151195,17.378188238,-17.5348765366,0.375603802,-0.0,0.0), 15.416540040627943)
  def test_GPS2POS(self):
    '''
    test GPS2POS function
    '''
    self.assertEqual(GPS2POS((52.516288,13.377689)), (-6.48982764810209, 9.159322471000536))
  pass
  
#--------------------- test with nose ---------------------------
def test_dist():
  '''unit test with nose'''
  assert dist(-7.83434151195,17.378188238,-17.5348765366,0.375603802,-0.0,0.0) == 15.416540040627943
  pass

def test_GPS2POS():
  '''unit test with nose'''
  assert GPS2POS((52.516288,13.377689)) == (-6.48982764810209, 9.159322471000536)
  pass


#--------------------- test with nose: decorator ---------------------------
_globals = {'tmpPoint':None}

def setup_test_GPS2POS():
  '''setup function call for running nose test'''
  _globals['tmpPoint'] = (52.516288,13.377689)
  pass

def teardown_test_GPS2POS():
  '''tear down function call for running nose test'''
  _globals['tmpPoint'] = None
  pass

@with_setup(setup_test_GPS2POS,teardown_test_GPS2POS)
def test():
  '''test function for executing nose test'''
  tmpPoint = _globals['tmpPoint']
  assert GPS2POS(tmpPoint) == (-6.48982764810209, 9.159322471000536)

if __name__ == '__main__':
  '''
  __name__ to make sure the function be excuted when used as a script not as a loaded module
  '''

  # doc test
  print "------------------- doctest -------------------"
  print "Run test with $python -m doctest -v solution_with_unittest.py\n"
  # unittest
  print "--------------- unittest  -----------------------"
  print "Run test with $python -m unittest -v solution_with_unittest\n"
  # nose
  print "--------------- nose  ---------------------------"
  print "Run test with $nosetests -v solution_with_unittest.py\n"


  # run actual code
  #find_her() 
  pass
