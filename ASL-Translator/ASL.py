import cv2
import numpy as np
import util as ut
import svm_train as st 
import re
from collections import Counter
from googletrans import Translator
translator = Translator()

####################################################################################################################
# from text import run

import os
import autocomplete
import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 120)

autocomplete.load()

def getNewChar(c):
   return c

def getPredictions(current_buffer, i):
   options = autocomplete.predict(current_buffer, '')[:4]
   finalOptions = [i[0] for i in options]
   print(' {}: \n'.format(current_buffer))
   return finalOptions

def printLatest(string):
   print(string)

def updateBuffer(buffer, c):
   return buffer + c

def speak(words):
	eng = ''
	for i in words:
		eng += i
		eng += ' '
	trans = translator.translate(eng, dest='hi')
	print(eng)
	print(trans.text)
	# print(trans)
	engine.say(words[-1])
	engine.runAndWait()

def run(c):
   os.system('clear')

   buffer = ''
   saved = ''
   sentence = []
   wordstring = ''

   i = 0

####################################################################################################################

model=st.trainSVM(17)
cam=int(input("Enter Camera number: "))
cap=cv2.VideoCapture(cam)
font = cv2.FONT_HERSHEY_SIMPLEX

def nothing(x) :
    pass

text= " "

temp=0
previouslabel=None
previousText=" "
label = None

arr = []
os.system('clear')

buffer = ''
saved = ''
sentence = []
wordstring = ''

i = 0

while(cap.isOpened()):
	
	_,img=cap.read()
	cv2.rectangle(img,(300,0),(600,300),(255,0,0),3) # bounding box which captures ASL sign to be detected by the system
	img1=img[0:300,300:600]
	img_ycrcb = cv2.cvtColor(img1, cv2.COLOR_BGR2YCR_CB)
	blur = cv2.GaussianBlur(img_ycrcb,(11,11),0)
	

	skin_ycrcb_min = np.array((0, 130, 103))
	skin_ycrcb_max = np.array((125, 152, 130))	
	
	mask = cv2.inRange(blur, skin_ycrcb_min, skin_ycrcb_max)  # detecting the hand in the bounding box using skin detection
	_,contours,hierarchy = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL, 2) 
	cnt=ut.getMaxContour(contours,2000)					  # using contours to capture the skin filtered image of the hand
	if (cnt.size!=0):
		gesture,label=ut.getGestureImg(cnt,img1,mask,model)   # passing the trained model for prediction and fetching the result
		
		cv2.putText(img,label,(300,150), font,7,(0,125,155),5)
		

		if(len(arr)==100):
			most_common,num_most_common = Counter(arr).most_common(1)[0]
			
			try:
				if(most_common=='Q'):
					most_common = ' '	
				elif(most_common=='F'):
					most_common = '4'
				elif(most_common=='M' or most_common=='N'):
					most_common = '3'
				elif(most_common=='K'):
					most_common = '2'
				# elif(most_common=='C'):
				# 	most_common = '1'
			except:
				continue

			print('selecting letter ' , most_common)
			arr=[]	

			newChar = most_common.lower()
		
			if(newChar.isalpha()):
				os.system('clear')
				buffer = updateBuffer(buffer, newChar)
				saved = updateBuffer(saved, newChar)
				printLatest(wordstring)
				i += 1
				suggestions = getPredictions(buffer, i)
				for word in suggestions:
					print(suggestions.index(word) + 1, word)

			if(newChar.isdigit()):
				wordstring += ' '
				wordstring += suggestions[int(newChar)-1]
				saved += ' '
				os.system('clear')
				sentence.append(suggestions[int(newChar)-1])
				trans = translator.translate(sentence, dest='hi')
				print(trans.text)
				print(trans)
				speak(sentence)
				buffer = ''

			if newChar == ' ':
				saved += ' '
				os.system('clear')
				sentence.append(buffer)
				wordstring += ' '
				wordstring += sentence[-1]
				speak(sentence)
				buffer = ''

		else:
			arr.append(label)

		
	        	
	
	img = cv2.flip( img, 1 ) 
	mask = cv2.flip( mask, 1 ) 
	
	
	cv2.imshow('Frame',img)
	cv2.imshow('Mask',mask)

	k = 0xFF & cv2.waitKey(10)
	if k == 27:
		break


cap.release()        
cv2.destroyAllWindows()
