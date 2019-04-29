from pprint import pprint
from Parser import Parser
import operator
import util
import os
import math
import sys

class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers. 
    A document is represented as a vector. Each dimension of the vector corresponds to a 
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """
    #Collection of document term vectors
    documentVectors=[]
    #Mapping of vector index to keyword
    vectorKeywordIndex=[]
    #Tidies terms
    parser=None

    idfvector=[]

    def __init__(self, documents=[], flag=""):
        self.documentVectors=[]
        self.parser = Parser()
        if(len(documents)>0):
            self.build(documents, flag)

    def build(self,documents, flag):
        """ Create the vector space for the passed document strings """
        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents)
        self.idfvector = [0] * len(self.vectorKeywordIndex)
        for (key, value) in self.vectorKeywordIndex.items():
            kinidf = sum(1 for document in documents if key in document)
            if kinidf > 0:
                self.idfvector[value] = float( math.log10( float( 2048/kinidf ) ) )
            else:
                self.idfvector[value] = 0
        self.documentVectors = [self.makeVector(document, flag) for document in documents]

    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """
        #Mapped documents into a single word string	
        vocabularyString = " ".join(documentList)

        vocabularyList = self.parser.tokenise(vocabularyString)
        #Remove common words which have no search value
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)

        vectorIndex={}
        offset=0
        #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        for word in uniqueVocabularyList:
            vectorIndex[word]=offset
            offset+=1
        return vectorIndex  #(keyword:position)
				
    def makeVector(self, wordString, flag):
        """ @pre: unique(vectorIndex) """
        #Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        wordList = self.parser.tokenise(wordString)
        wordList = self.parser.removeStopWords(wordList)
        for word in wordList:
            try:
                vector[self.vectorKeywordIndex[word]] += 1 #Use simple Term Count Model
            except:
                continue
        if flag == "tfidf":
            for index in range(0,len(vector)):
                vector[index] = float( vector[index] * self.idfvector[index] ) #Convert to TF-IDF Model
        return vector

    def buildQueryVector(self, termList, flag):
        query = self.makeVector(" ".join(termList), flag)
        if query == [0] * len(self.vectorKeywordIndex):
            print("\nThere is no related document in the corpus. Sorry.")
            sys.exit()
        return query

    def search(self, searchList, compare, flag): 
        queryVector = self.buildQueryVector(searchList, flag)
        if compare == "cos":
            ratings = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors]
        elif compare == "dis":
            ratings = [util.Euclidean(queryVector, documentVector) for documentVector in self.documentVectors]
        return ratings

    def feedbacksearch(self, searchList, wordString, flag):
        queryVector = self.buildQueryVector(searchList, flag)
        feedback = self.makeVector(wordString, flag)
        for index in range( 0, len(queryVector)):
            queryVector[index] = float( queryVector[index] + feedback[index] * (1/2) )
        ratings = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors]
        return ratings

##########################################################################################################################################################
def printAnswer(data, Dict):
	print("DocID   Score")
	checked = []
	index = 0
	while index < 5:
		for (key, value) in Dict.items():
			if (value == data[index]) and (key not in checked):
				print( key.replace(".product", ""), value, sep="  " )
				checked.append(key)
				break
		index = index + 1

# Main Function p.s. python version: "python 3.6"
path = "Documents"
documents = []
files = os.listdir(path)
for file in files:
	f = open( path + "/" + file ); 
	iter_f = iter(f); 
	str = ""
	for line in iter_f:
		str = str + line
		documents.append(str)

# Input Query
Query = []
Query = input("Please input query: \n")
print("\nThe time for calculating will be about 2-3 minutes, please be patient to wait. Thank you!\n")
# TF + Cosine Similarity ###########################################################################################################
vectorSpaceTF = VectorSpace(documents, "tf") #print(vectorSpaceTF.vectorKeywordIndex) #print(vectorSpaceTF.documentVectors)
data = vectorSpaceTF.search([Query], "cos", "tf") #print(data)
Dict = {} # dict for record every score of *.product
for file,num in zip(files,data):
	Dict.update({file: num})
data.sort(reverse=True) #print( data )
print( "TF Weighting + Cosine Similarity: " )
printAnswer( data, Dict)
	
# TF + Euclidean Distance #########################################################################################################
data = vectorSpaceTF.search([Query], "dis", "tf")
Dict = {} # dict for record every score of *.product
for file,num in zip(files,data):
	Dict.update({file: num})
data.sort(reverse=False)
print( "\nTF Weighting + Euclidean Distance: " )
printAnswer( data, Dict)

# TF-IDF + Cosine Similarity ######################################################################################################
vectorSpaceTFIDF = VectorSpace(documents, "tfidf")
data = vectorSpaceTFIDF.search([Query], "cos", "tfidf")
Dict = {}
for file,num in zip(files,data):
	Dict.update({file: num})
data.sort(reverse=True) #print( data )
print( "\nTF-IDF Weighting + Cosine Similarity: " )
for (key, value) in Dict.items():
	if value == data[0] :
		FeedbackQuery = key
		break
printAnswer( data, Dict)

# TF-IDF + Euclidean Distance #####################################################################################################
data = vectorSpaceTFIDF.search([Query], "dis", "tfidf")
Dict = {}
for file,num in zip(files,data):
	Dict.update({file: num})
data.sort(reverse=False)
print( "\nTF-IDF Weighting + Euclidean Distance: " )
printAnswer( data, Dict)

# Feedback Queries + TF-IDF Weighting + Cosine Similarity #########################################################################
f = open( path + "/" + FeedbackQuery )
iter_f = iter(f)
str = ""
for line in iter_f:
	str = str + line
data = vectorSpaceTFIDF.feedbacksearch([Query], str, "tfidf")
Dict = {}
for file,num in zip(files,data):
	Dict.update({file: num})
data.sort(reverse=True)
print( "\nFeedback Queries + TF-IDF Weighting + Cosine Similarity: " )
printAnswer( data, Dict)

