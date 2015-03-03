import multiprocessing
import timeit
from vsm import *
from time import gmtime, strftime
from vsm.corpus import Corpus
from vsm.model.ldacgsmulti import LdaCgsMulti as LDA
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
c = Corpus.load("../TrainingScript/corpus/sepEntries-nltk3.npz")

def test(cores):
	m = LDA(c, 'article', 20) #initialize model
    	#test on n_cores, doesn't have to be many iterations
   	m.train( n_iterations=5,n_proc=cores)

def plot(time):
	pyplot.title("VSM Performance on SEP Corpus")
	pyplot.plot(time.keys(), time.values())
	pyplot.xlabel("# cores")
	pyplot.ylabel("time (s)")
	pyplot.savefig('graph.png')
if __name__ == '__main__':
	times = dict();
	startTime = strftime("%Y-%m-%d %H:%M:%S", gmtime())
	for cores in range(1,multiprocessing.cpu_count()+1): 
		times[cores] = timeit.repeat("test(cores)", setup="from __main__ import test; cores={cores}".format(cores=cores), repeat=3 ,number=1)
	print "Start Time :" + startTime
	print "End Time :"+strftime("%Y-%m-%d %H:%M:%S", gmtime())
	plot(times)
