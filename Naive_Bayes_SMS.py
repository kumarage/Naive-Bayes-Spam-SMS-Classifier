# Based on Siraj Raval llSourcell  and AlanBuzdar code https://www.youtube.com/watch?v=PrkiRVcrxOs 
# Code: https://github.com/llSourcell/naive_bayes_classifier/blob/master/naive_bayes.py
# creating a sms spam model

import numpy as np
import pandas as pd
import csv


trainPos = {}
trainNeg = {}
positiveTotal = 0
negativeTotal = 0

def train(filename):
	total=0
	numspam=0
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader) #skipping column names
		for row in csvFileReader:
			if row[0]=='spam':
				numspam += 1
			total += 1
			process_body(row[0],row[1])

	csvfile.close()
	pA = numspam/float(total)
	pnA = 1-pA
	return pA, pnA	

#reading words from a specific sms
def process_body(label,body):
	body = body.split()
	global positiveTotal
	global negativeTotal

	for word in body:
		if label == 'spam':
			trainPos[word] = trainPos.get(word,0) + 1
			positiveTotal += 1
		else:
			trainNeg[word] = trainNeg.get(word,0) + 1
			negativeTotal += 1

#classifies a new sms as spam or notnspam
def classify(sms,pA,pnA):
	spamY = pA*pconditionalSms(sms, True) #this calculates P(A|B)
	spamN = pnA*pconditionalSms(sms, False) #this calculates P(~A|B)

	return spamY > spamN

def pconditionalSms(body, stat):
	result = 1.0
	numwords = len(body.split())
	for word in body:
		result *= pconditionalWord(body, stat, numwords)
	return result

#Multinomial Naive Bayes with Laplace smoothing alpha = 1, if Lindstone smoothing alpha < 1
#gives the conditional probability p(B-i|A) with Laplace smoothing

def pconditionalWord(word,spam,numwords):
	alpha = 1.0
	if spam:
		return (trainPos.get(word,0)+alpha)/(float)(positiveTotal+alpha*numwords)
	return (trainNeg.get(word,0)+alpha)/(float)(negativeTotal+alpha*numwords)


#main program
pA,pnA=train('spam_train.csv')
print(pA,pnA)
print(positiveTotal,negativeTotal)
testVal = []
predVal = []

#testing with the testing sms dataset
testfile='spam_test.csv'
with open(testfile,'r') as csvtest:
	csvTestReader = csv.reader(csvtest)
	next(csvTestReader) #skipping column names

	for row in csvTestReader:
		stat = classify(row[1],pA,pnA)
		#print (stat)
		num=0
		if stat == 'True':
			val='spam'
		else:
			val='ham'
		testVal.append(val)
		predVal.append(row[0])

csvtest.close()
#concatonate two lists into one array
listfinal=np.array((testVal,predVal))
listfinal=listfinal.transpose()
#number of misclassifications occured
n_missed=(listfinal[:,0] != listfinal[:,1]).sum()
#total number of test samples
total = listfinal.shape[0]
# performace of the classifier
print('Naive Bayes Classifier predicts with %f%% accuracy' % ((1-n_missed/total)*100))
