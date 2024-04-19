# Cipher Decoding

This notebook recreates a famous cryptography paper written by Coram and Beineke. I use the Metropolis-Hastings algorithm to crack a simple "substitution cipher." To encode a message with a substitution cipher, one simply swaps each letter of the real message with a corresponding code letter. For example, a substitution cipher might pick the letter 'd' to represent the letter 'a'. So, to encode the real message, one simply changes every 'a' in the real message to 'd'. To create a seemingly uncrackable code, you would swap every letter of the alphabet for another, not just 'a' to 'd'. To decode a message, one must recover the correct swaps between code and real letters--the key.

The algorithm recovers the key as follows. First, initialize a random key. Then propose a new key by changing two letters *in code space* uniformly at random (this is shown more clearly in the program below). The new key is accepted or rejected with a given probability consistent with the classic Metropolis-Hastings algorithm. Iterate the algorithm for a given number of steps, and in all likelihood, the algorithm has recovered the correct key, or something very close.

In order to use the Metropolis-Hastings algorithm, though, we need to know the likelihood of a message decoded with a guess to the correct key. We measure the likelihood of a key by assessing its consistency with some known rules of language--for example, how often the letter 'b' follows the letter 't'. I downloaded the book War and Peace and recorded exactly that for all letters. This yields a matrix of first order transition probabilities, i.e. how often each letter of the alphabet follows another.

At each iteration of the algorithm, we decode the cipher given our current guess to the key, and then multiply the corresponding transition probabilities together. The product measures the likelihood of the current key. The algorithm navigates the state space of substitution ciphers efficiently, finding the most likely ciphers given the observed coded message. We write a program to solve substitution ciphers and demonstrate an example of a coded and decoded message below (one of my favorite quotes about probability). 


```python
## Stanford MCMC decoding

import numpy as np
import matplotlib.pyplot as plt

## Idea is to first get some sort of prior distribution for how likely characters are to follow each other
## If you are working with a simple substitution cipher i.e. one character = another character, one to one correspondence

# So, need to download some massive text document, and record how likely it is for one character to follow another
# Need program to record the counts and put this into a transition matrix

filepath = 'warAndPeace.txt'

# Get rid of characters we don't want to use

charactersToReplace = ['"', "'", '--', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', \
    '-', '_', '=', '+', '[', ']', '{', '}', ';', ':', ',', '.', '/', '?', '<', '>', '|', \
        '\n', '\t', 'à', 'ä', 'é', 'ê']

with open(filepath, 'r') as file:

    text = file.read().lower()
    text = ''.join(' ' if char in charactersToReplace else char for char in text)
    text = ''.join(char for char in text if not char.isdigit())

textArray = np.array(list(text))
uniqueCharacters = np.unique(textArray)

# Now that we have removed all of the unwanted characters from the text, we can count the transitions

epsilon = .00001
transitionMatrix = np.zeros([len(uniqueCharacters), len(uniqueCharacters)]) + epsilon
numCharacters = 3000000

for charIndex in range(numCharacters):

    currentChar = text[charIndex]
    nextChar = text[charIndex + 1]
    tempTransitionRow = np.where(currentChar == uniqueCharacters)
    tempTransitionCol = np.where(nextChar == uniqueCharacters)
    transitionMatrix[tempTransitionRow, tempTransitionCol] += 1

# Normalizing

for iiRow in range(np.shape(transitionMatrix)[0]):

    transitionMatrix[iiRow, :] = transitionMatrix[iiRow, :] / np.sum(transitionMatrix[iiRow, :])

```


```python
## Now that we have the transition matrix, we can calculate the plausibility/likelihood of each coded text
# First create a simple substitution cipher

import random
import numpy as np
import sys 

alphabet = ' abcdefghijklmnopqrstuvwxyz'

def makeRandomKey():

    tempList = list(alphabet)
    random.shuffle(tempList)
    randomKey = dict(zip(alphabet, tempList))

    return randomKey

filepath = 'dice_fair_success.txt'
with open(filepath, 'r') as file:

    textString = file.read().lower()
    textString = ''.join(' ' if char in charactersToReplace else char for char in textString) # Replace characters with spaces using a generator expression
    textString = ''.join(char for char in textString if not char.isdigit())

cipherKey = makeRandomKey()

# Use the cipher on the string
cipher = []

for charIndex in range(len(textString)):

    currentChar = textString[charIndex]
    tempTargetChar = cipherKey[currentChar]
    cipher.append(tempTargetChar)

cipher = ''.join(cipher)

## Using your deciphering method for a known cipher

numChars = len(cipher)
decodedCipher = []

for jjChar in range(numChars):

    tempChar = cipher[jjChar]
    tempDecodedLetter = [key for key, value in cipherKey.items() if value == tempChar][0]
    decodedCipher.append(tempDecodedLetter[0])

decodedCipher = ''.join(decodedCipher)
print('Coded message: ', cipher, '\n')
print('Decoded message: ', decodedCipher)
machineEpsilon = sys.float_info.epsilon

```

    Coded message:  owcnhcnhqlmhqehf hergqdcomhtuqomnhrzquohkdqzrzcico hhcehaqjhwrnhfrjmhowmhsqdijhrhkmdempohfmpwrlcnfhhwmhwrnhrohimrnohpqlpmjmjhnqhfupwhoqhqudhcfkmdempohclomiimponhowrohclhqdjmdhoqhkdmjcpohicooimhkrdonhqehcohhhsmhlmmjhlqohnqigmhcllufmdrzimhjceemdmlocrihmturocqlnhhzuohprlhunmhjcpmhscowhercdhnuppmnnhh 
    
    Decoded message:  this is one of my favorite quotes about probability  if god has made the world a perfect mechanism  he has at least conceded so much to our imperfect intellects that in order to predict little parts of it   we need not solve innumerable differential equations  but can use dice with fair success  



```python
def proposeNewKey(key):

    pair = random.sample(alphabet, 2)
    newKey = dict(key)
    newKey[pair[0]] = key[pair[1]]
    newKey[pair[1]] = key[pair[0]]

    return newKey
```


```python
## Now you can use the algorithm to start decoding

numIterations = 26000
numChars = len(cipher)
codeSpace = makeRandomKey()
accept = np.zeros([numIterations])
acceptanceProbabilities = np.zeros([numIterations])
plausibility = np.zeros([numIterations])

for iiIteration in range(numIterations):

    tempTransitionProbs = np.zeros([numChars - 1, 1])

    for jjChar in range(numChars - 1):

        currentChar = cipher[jjChar]
        nextChar = cipher[jjChar + 1]
        rowIndex = np.where(uniqueCharacters == [key for key, value in codeSpace.items() if value == currentChar])[0][0]
        columnIndex = np.where(uniqueCharacters == [key for key, value in codeSpace.items() if value == nextChar])[0][0]
        tempTransitionProbs[jjChar] = transitionMatrix[rowIndex, columnIndex]

    for jjProb in range(np.size(tempTransitionProbs)):

        if tempTransitionProbs[jjProb] < machineEpsilon:

            tempTransitionProbs[jjProb] = machineEpsilon

    currentPlausibility = np.sum(np.log(tempTransitionProbs))
    plausibility[iiIteration] = currentPlausibility

    candidateCodeSpace = proposeNewKey(codeSpace)
    candidateTransitionProbs = np.zeros([numChars - 1, 1])

    for jjChar in range(numChars - 1):

        currentChar = cipher[jjChar]
        nextChar = cipher[jjChar + 1]
        rowIndex = np.where(uniqueCharacters == [key for key, value in candidateCodeSpace.items() if value == currentChar])[0][0]
        columnIndex = np.where(uniqueCharacters == [key for key, value in candidateCodeSpace.items() if value == nextChar])[0][0]
        candidateTransitionProbs[jjChar] = transitionMatrix[rowIndex, columnIndex]

    for jjProb in range(np.size(tempTransitionProbs)):

        if candidateTransitionProbs[jjProb] < machineEpsilon:

            candidateTransitionProbs[jjProb] = machineEpsilon

    candidatePlausibility = np.sum(np.log(candidateTransitionProbs))
    logDifference = candidatePlausibility - currentPlausibility
    
    if logDifference >= 0:

        acceptanceProb = 1

    else:

        acceptanceProb = np.exp(logDifference)

    u = np.random.uniform()
    acceptanceProbabilities[iiIteration] = acceptanceProb

    if u < acceptanceProb:

        accept[iiIteration] = 1
        codeSpace = candidateCodeSpace

    # Decode the message

    if np.mod(iiIteration, 2000) == 0:

        mcmcdecodedCipher = []

        for jjChar in range(numChars):

            tempChar = cipher[jjChar]
            tempDecodedLetter = [key for key, value in codeSpace.items() if value == tempChar][0]
            mcmcdecodedCipher.append(tempDecodedLetter[0])

        mcmcdecodedCipher = ''.join(mcmcdecodedCipher)
        print('Iteration: ', iiIteration, '---', mcmcdecodedCipher, '\n')



```

    Iteration:  0 --- hlotmotmrwdmrfmjzmfiprxohdmsgrhdtminrghmexrninoaohzmmofmbrqmlitmjiqdmhldmkrxaqmimedxfduhmjduliwotjmmldmlitmihmadithmurwudqdqmtrmjgulmhrmrgxmojedxfduhmowhdaaduhtmhlihmowmrxqdxmhrmexdqouhmaohhadmeixhtmrfmohmmmkdmwddqmwrhmtrapdmowwgjdxinadmqoffdxdwhoiamdsgihorwtmmnghmuiwmgtdmqoudmkohlmfioxmtguudttmm 
    
    Iteration:  2000 --- this is one of my favorite quotes apout bropapility  if kod has made the world a berfect mechanism  he has at least conceded so much to our imberfect intellects that in order to bredict little barts of it   we need not solve innumeraple differential equations  put can use dice with fair success   
    
    Iteration:  4000 --- this is one of my favorite quotes agout progagility  if bod has made the world a perfect mechanism  he has at least conceded so much to our imperfect intellects that in order to predict little parts of it   we need not solve innumeragle differential equations  gut can use dice with fair success   
    
    Iteration:  6000 --- this is one of my fakorite quotes apout bropapility  if god has made the world a berfect mechanism  he has at least conceded so much to our imberfect intellects that in order to bredict little barts of it   we need not solke innumeraple differential equations  put can use dice with fair success   
    
    Iteration:  8000 --- this is one of my fakorite quotes apout bropapility  if jod has made the world a berfect mechanism  he has at least conceded so much to our imberfect intellects that in order to bredict little barts of it   we need not solke innumeraple differential equations  put can use dice with fair success   
    
    Iteration:  10000 --- this is one of my fakorite quotes about probability  if vod has made the world a perfect mechanism  he has at least conceded so much to our imperfect intellects that in order to predict little parts of it   we need not solke innumerable differential equations  but can use dice with fair success   
    
    Iteration:  12000 --- this is one of my favorite quotes about probability  if god has made the world a perfect mechanism  he has at least conceded so much to our imperfect intellects that in order to predict little parts of it   we need not solve innumerable differential equations  but can use dice with fair success   
    
    Iteration:  14000 --- this is one of my favorite quotes about probability  if jod has made the world a perfect mechanism  he has at least conceded so much to our imperfect intellects that in order to predict little parts of it   we need not solve innumerable differential equations  but can use dice with fair success   
    
    Iteration:  16000 --- this is one of my favorite quotes about probability  if god has made the world a perfect mechanism  he has at least conceded so much to our imperfect intellects that in order to predict little parts of it   we need not solve innumerable differential equations  but can use dice with fair success   
    
    Iteration:  18000 --- this is one of my favorite quotes about probability  if jod has made the world a perfect mechanism  he has at least conceded so much to our imperfect intellects that in order to predict little parts of it   we need not solve innumerable differential equations  but can use dice with fair success   
    
    Iteration:  20000 --- this is one of my favorite quotes apout bropapility  if god has made the world a berfect mechanism  he has at least conceded so much to our imberfect intellects that in order to bredict little barts of it   we need not solve innumeraple differential equations  put can use dice with fair success   
    
    Iteration:  22000 --- this is one of my favorite quotes about probability  if god has made the world a perfect mechanism  he has at least conceded so much to our imperfect intellects that in order to predict little parts of it   we need not solve innumerable differential equations  but can use dice with fair success   
    
    Iteration:  24000 --- this is one of my favorite juotes about probability  if god has made the world a perfect mechanism  he has at least conceded so much to our imperfect intellects that in order to predict little parts of it   we need not solve innumerable differential ejuations  but can use dice with fair success   
    



```python
numChars = len(cipher)
knowndecodedCipher = []

for jjChar in range(numChars):

    tempChar = cipher[jjChar]
    tempDecodedLetter = [key for key, value in cipherKey.items() if value == tempChar][0]
    knowndecodedCipher.append(tempDecodedLetter[0])

knowndecodedCipher = ''.join(knowndecodedCipher)

print('Decoded with MCMC: ', mcmcdecodedCipher)
print('Known decoded message: ', knowndecodedCipher)
```

    Decoded with MCMC:  this is one of my favorite juotes about probability  if god has made the world a perfect mechanism  he has at least conceded so much to our imperfect intellects that in order to predict little parts of it   we need not solve innumerable differential ejuations  but can use dice with fair success  
    Known decoded message:  this is one of my favorite quotes about probability  if god has made the world a perfect mechanism  he has at least conceded so much to our imperfect intellects that in order to predict little parts of it   we need not solve innumerable differential equations  but can use dice with fair success  


We can see that the MCMC method has nearly perfectly decoded the substitution cipher. In fact, we see in the second to last printed iteration (iteration 22000) that it was decoded perfectly. We plot the likelihood of the current decoding function across iterations of the algorithm. The likelihood makes some large jumps at the start of the algorithm but then only makes minor changes to the decoding key. The decoded messages printed during the algorithm reflect this above. 


```python
# Plausibility 

plt.plot(list(range(numIterations)), plausibility, color = 'r')
plt.xlabel('Iteration')
plt.ylabel('Likelihood')
plt.title('Likelihood of Current Decryption Key over Time')
plt.show()
```


    
![png](README_files/README_7_0.png)
    

