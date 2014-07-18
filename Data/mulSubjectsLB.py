import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sci


data1 = json.load(open('s11-gammasys-home-impaired.json','r'))
letterb1 = data1[6]



data2 = json.load(open('s20-gammasys-gifford-unimpaired.json','r'))
letterb2 = data2[6]

eeg1 = np.array(letterb1['eeg']['trial 1']).T
eeg2 = np.array(letterb2['eeg']['trial 1']).T
print eeg1.shape
#Collect all of the segments from the letter-b trial
starts1 = np.where(np.diff(np.abs(eeg1[:,-1])) > 0)[0]
stimuli1 = [chr(int(n)) for n in np.abs(eeg1[starts1+1,-1])]
targetLetter1 = letterb1['protocol'][-1]
targetSegments1 = np.array(stimuli1) == targetLetter1

starts2 = np.where(np.diff(np.abs(eeg2[:,-1])) > 0)[0]
stimuli2 = [chr(int(n)) for n in np.abs(eeg2[starts2+1,-1])]
targetLetter2 = letterb2['protocol'][-1]
targetSegments2 = np.array(stimuli2) == targetLetter2

##########################################################################3
for i,(s,stim,targ) in enumerate(zip(starts1,stimuli1,targetSegments1)):
   print s,stim,targ,'; ',
   if (i+1) % 5 == 0:
       print

print "window lengths ",np.diff(starts1)
print "minimum length ",np.min(np.diff(starts1))

indices1 = np.array([ np.arange(s,s+210) for s in starts1 ])
print "indices shape ",indices1.shape

segments1 = eeg1[indices1,:8]
print "sigment shape ",segments1.shape

targetSegsp3 = segments1[targetSegments1,:,4]
Twithclassp31 = np.hstack((np.ones((targetSegsp3.shape[0], 1), dtype=targetSegsp3.dtype),targetSegsp3))
nontargetSegsp3 = segments1[targetSegments1==False,:,4]
NTwithclassp31 = np.hstack((np.zeros((nontargetSegsp3.shape[0], 1), dtype=nontargetSegsp3.dtype),nontargetSegsp3))
print "P4, target segment shape", targetSegsp3.shape
print "P4, nontarget segment shape", nontargetSegsp3.shape
classDatap31=np.vstack((Twithclassp31,NTwithclassp31))
classDatap31=np.hstack((np.ones((classDatap31.shape[0], 1)),classDatap31))

targetSegsp4 = segments1[targetSegments1,:,5]
Twithclassp41 = np.hstack((np.ones((targetSegsp4.shape[0], 1), dtype=targetSegsp4.dtype),targetSegsp4))
nontargetSegsp4 = segments1[targetSegments1==False,:,5]
NTwithclassp41 = np.hstack((np.zeros((nontargetSegsp4.shape[0], 1), dtype=nontargetSegsp4.dtype),nontargetSegsp4))
print "P4, target segment shape", targetSegsp4.shape
print "P4, nontarget segment shape", nontargetSegsp4.shape
classDatap41=np.vstack((Twithclassp41,NTwithclassp41))
classDatap41=np.hstack((np.ones((classDatap41.shape[0], 1)),classDatap41))

P3data = segments1[:,:,4]
#print P3data
P4data = segments1[:,:,5]
#print "P4, segment shape", P3data.shape

sci.savemat('classDataP3s11LBsub.mat', mdict = {'classData':classDatap31})
sci.savemat('classDataP4s11LBsub.mat', mdict = {'classData':classDatap41})

#####################################################################

for i,(s,stim,targ) in enumerate(zip(starts2,stimuli2,targetSegments2)):
   print s,stim,targ,'; ',
   if (i+1) % 5 == 0:
       print

print "window lengths ",np.diff(starts2)
print "minimum length ",np.min(np.diff(starts2))

indices2 = np.array([ np.arange(s,s+210) for s in starts2 ])
print "indices shape ",indices2.shape

segments2 = eeg2[indices2,:8]
print "sigment shape ",segments2.shape

targetSegsp3 = segments2[targetSegments2,:,4]
Twithclassp32 = np.hstack((np.ones((targetSegsp3.shape[0], 1), dtype=targetSegsp3.dtype),targetSegsp3))
nontargetSegsp3 = segments2[targetSegments2==False,:,4]
NTwithclassp32 = np.hstack((np.zeros((nontargetSegsp3.shape[0], 1), dtype=nontargetSegsp3.dtype),nontargetSegsp3))
print "P4, target segment shape", targetSegsp3.shape
print "P4, nontarget segment shape", nontargetSegsp3.shape
classDatap32=np.vstack((Twithclassp32,NTwithclassp32))
classDatap32=np.hstack((2*np.ones((classDatap32.shape[0], 1)),classDatap32))

targetSegsp4 = segments2[targetSegments2,:,5]
Twithclassp42 = np.hstack((np.ones((targetSegsp4.shape[0], 1), dtype=targetSegsp4.dtype),targetSegsp4))
nontargetSegsp4 = segments2[targetSegments2==False,:,5]
NTwithclassp42 = np.hstack((np.zeros((nontargetSegsp4.shape[0], 1), dtype=nontargetSegsp4.dtype),nontargetSegsp4))
print "P4, target segment shape", targetSegsp3.shape
print "P4, nontarget segment shape", nontargetSegsp3.shape
classDatap42=np.vstack((Twithclassp42,NTwithclassp42))
classDatap42=np.hstack((2*np.ones((classDatap42.shape[0], 1)),classDatap42))

P3data = segments2[:,:,4]
#print P3data
P4data = segments2[:,:,5]
#print "P4, segment shape", P3data.shape

sci.savemat('classDataP3s20LBsub.mat', mdict = {'classData':classDatap32})
sci.savemat('classDataP4s20LBsub.mat', mdict = {'classData':classDatap42})
allsubP4 = np.vstack((classDatap41,classDatap42))
allsubP3 = np.vstack((classDatap31,classDatap32))

sci.savemat('allSubLBP3.mat', mdict = {'classData':allsubP3})
sci.savemat('allSubLBP4.mat', mdict = {'classData':allsubP4})



