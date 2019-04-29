import sys
#http://www.scipy.org/
from scipy.spatial import distance

try:
	from numpy import dot
	from numpy.linalg import norm
except:
	print("Error: Requires numpy from http://www.scipy.org/. Have you installed scipy?")
	sys.exit() 

def removeDuplicates(list):
	""" remove duplicates from a list """
	return set((item for item in list))

def Euclidean(vector1, vector2):
    if(vector1 != [0]*len(vector1)) and (vector2 != [0]*len(vector2)):
        dist = 0
        for a,b in zip(vector1,vector2):
            dist = dist + ((a-b)**2)
        return round( float(dist**(1/2)), 6)
    else:
        return sys.maxsize

def cosine(vector1, vector2):
	""" related documents j and q are in the concept space by comparing the vectors : cosine  = ( V1 * V2 ) / ||V1|| x ||V2|| """
	try:
		if(vector1 != [0]*len(vector1)) and (vector2 != [0]*len(vector2)):
			return round(float(dot(vector1,vector2) / (norm(vector1) * norm(vector2))), 6)
	except:
			return 0
