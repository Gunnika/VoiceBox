import os
# import getch
import autocomplete
import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 120)

autocomplete.load()

def getNewChar(c):
   return c


def getPredictions(current_buffer, i):
   options = autocomplete.predict(current_buffer, '')[:5]
   finalOptions = [i[0] for i in options]
   print(' {}: \n'.format(current_buffer))
   return finalOptions


def printLatest(string):
   print(string)


def updateBuffer(buffer, c):
   return buffer + c


def speak(words):
   engine.say(words[-1])
   engine.runAndWait()


def run(c):
   os.system('clear')

   buffer = ''
   saved = ''
   sentence = []
   wordstring = ''

   i = 0

   while(1):
      #  newChar = getNewChar(getch.getch())
      newChar = c

      if(newChar.isalpha()):
         os.system('clear')
         buffer = updateBuffer(buffer, newChar)
         saved = updateBuffer(saved, newChar)
         # saved += sentence[-1]
      #    printLatest(saved)
         printLatest(wordstring)
         i += 1
         suggestions = getPredictions(buffer, i)
         # print(suggestions, end='\r')
         for word in suggestions:
            print(suggestions.index(word) + 1, word)

      if(newChar.isdigit()):
         wordstring += ' '
         wordstring += suggestions[int(newChar)-1]
         saved += ' '
      #    saved += suggestions[int(newChar)-1]
         os.system('clear')
         sentence.append(suggestions[int(newChar)-1])

         # print(suggestions[int(newChar)-1])
         speak(sentence)
         buffer = ''

      if newChar == ' ':
         saved += ' '
         os.system('clear')
         sentence.append(buffer)
         wordstring += ' '
         wordstring += sentence[-1]

         # print(buffer)
         speak(sentence)
         buffer = ''


