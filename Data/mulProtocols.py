import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sci


data1 = json.load(open('s11-gammasys-home-impaired.json','r'))
letterb1 = data1[6]
letterp1 = data1[4]
letterd1 = data1[5]


data2 = json.load(open('s20-gammasys-gifford-unimpaired.json','r'))
letterb2 = data2[6]
letterp2 = data2[4]
letterd2 = data2[5]

eeg1b = np.array(letterb1['eeg']['trial 1']).T
eeg1p = np.array(letterp1['eeg']['trial 1']).T
eeg1d = np.array(letterd1['eeg']['trial 1']).T

eeg2b = np.array(letterb2['eeg']['trial 1']).T
eeg2p = np.array(letterp2['eeg']['trial 1']).T
eeg2d = np.array(letterd2['eeg']['trial 1']).T

#Collect all of the segments from the letter-b trial
starts1b = np.where(np.diff(np.abs(eeg1b[:,-1])) > 0)[0]
stimuli1b = [chr(int(n)) for n in np.abs(eeg1b[starts1b+1,-1])]
targetLetter1b = letterb1['protocol'][-1]
targetSegments1b = np.array(stimuli1b) == targetLetter1b

starts2b = np.where(np.diff(np.abs(eeg2b[:,-1])) > 0)[0]
stimuli2b = [chr(int(n)) for n in np.abs(eeg2b[starts2b+1,-1])]
targetLetter2b = letterb2['protocol'][-1]
targetSegments2b = np.array(stimuli2b) == targetLetter2b

#Collect all of the segments from the letter-p trial
starts1p = np.where(np.diff(np.abs(eeg1p[:,-1])) > 0)[0]
stimuli1p = [chr(int(n)) for n in np.abs(eeg1p[starts1p+1,-1])]
targetLetter1p = letterp1['protocol'][-1]
targetSegments1p = np.array(stimuli1p) == targetLetter1p

starts2p = np.where(np.diff(np.abs(eeg2p[:,-1])) > 0)[0]
stimuli2p = [chr(int(n)) for n in np.abs(eeg2p[starts2p+1,-1])]
targetLetter2p = letterp2['protocol'][-1]
targetSegments2p = np.array(stimuli2p) == targetLetter2p

#Collect all of the segments from the letter-d trial
starts1d = np.where(np.diff(np.abs(eeg1d[:,-1])) > 0)[0]
stimuli1d = [chr(int(n)) for n in np.abs(eeg1d[starts1d+1,-1])]
targetLetter1d = letterd1['protocol'][-1]
targetSegments1d = np.array(stimuli1d) == targetLetter1d

starts2d = np.where(np.diff(np.abs(eeg2d[:,-1])) > 0)[0]
stimuli2d = [chr(int(n)) for n in np.abs(eeg2d[starts2d+1,-1])]
targetLetter2d = letterd2['protocol'][-1]
targetSegments2d = np.array(stimuli2d) == targetLetter2d

########################################SUBJECT 1#############################################

for i,(s,stim,targ) in enumerate(zip(starts1b,stimuli1b,targetSegments1b)):
   print s,stim,targ,'; ',
   if (i+1) % 5 == 0:
       print

indices1b = np.array([ np.arange(s,s+210) for s in starts1b ])
segments1b = eeg1b[indices1b,:8]
P4data41b = segments1b[:,:,5]

for i,(s,stim,targ) in enumerate(zip(starts1p,stimuli1p,targetSegments1p)):
   print s,stim,targ,'; ',
   if (i+1) % 5 == 0:
       print

indices1p = np.array([ np.arange(s,s+210) for s in starts1p ])
segments1p = eeg1p[indices1p,:8]
P4data41p = segments1p[:,:,5]

for i,(s,stim,targ) in enumerate(zip(starts1d,stimuli1d,targetSegments1d)):
   print s,stim,targ,'; ',
   if (i+1) % 5 == 0:
       print

indices1d = np.array([ np.arange(s,s+210) for s in starts1d ])
segments1d = eeg1d[indices1d,:8]
P4data41d = segments1d[:,:,5]

targetSegsp3b = segments1b[targetSegments1b,:,4]
Twithclassp31b = np.hstack((np.ones((targetSegsp3b.shape[0], 1), dtype=targetSegsp3b.dtype),targetSegsp3b))
nontargetSegsp3b = segments1b[targetSegments1b==False,:,4]
NTwithclassp31b = np.hstack((np.zeros((nontargetSegsp3b.shape[0], 1), dtype=nontargetSegsp3b.dtype),nontargetSegsp3b))

targetSegsp3p = segments1p[targetSegments1p,:,4]
Twithclassp31p = np.hstack((np.ones((targetSegsp3p.shape[0], 1), dtype=targetSegsp3p.dtype),targetSegsp3p))
nontargetSegsp3p = segments1p[targetSegments1p==False,:,4]
NTwithclassp31p = np.hstack((np.zeros((nontargetSegsp3p.shape[0], 1), dtype=nontargetSegsp3p.dtype),nontargetSegsp3p))

targetSegsp3d = segments1d[targetSegments1d,:,4]
Twithclassp31d = np.hstack((np.ones((targetSegsp3d.shape[0], 1), dtype=targetSegsp3d.dtype),targetSegsp3d))
nontargetSegsp3d = segments1d[targetSegments1d==False,:,4]
NTwithclassp31d = np.hstack((np.zeros((nontargetSegsp3d.shape[0], 1), dtype=nontargetSegsp3d.dtype),nontargetSegsp3d))

classDatap31=np.vstack((Twithclassp31b,Twithclassp31p,Twithclassp31d,NTwithclassp31b,NTwithclassp31p,NTwithclassp31d))
classDatap31=np.hstack((np.ones((classDatap31.shape[0], 1)),classDatap31))
###################################################
targetSegsp4b = segments1b[targetSegments1b,:,5]
Twithclassp41b = np.hstack((np.ones((targetSegsp4b.shape[0], 1), dtype=targetSegsp4b.dtype),targetSegsp4b))
nontargetSegsp4b = segments1b[targetSegments1b==False,:,5]
NTwithclassp41b = np.hstack((np.zeros((nontargetSegsp4b.shape[0], 1), dtype=nontargetSegsp4b.dtype),nontargetSegsp4b))

targetSegsp4p = segments1p[targetSegments1p,:,5]
Twithclassp41p = np.hstack((np.ones((targetSegsp4p.shape[0], 1), dtype=targetSegsp4p.dtype)*2,targetSegsp4p))
nontargetSegsp4p = segments1p[targetSegments1p==False,:,5]
NTwithclassp41p = np.hstack((np.zeros((nontargetSegsp4p.shape[0], 1), dtype=nontargetSegsp4p.dtype)-1,nontargetSegsp4p))

targetSegsp4d = segments1d[targetSegments1d,:,5]
Twithclassp41d = np.hstack((np.ones((targetSegsp4d.shape[0], 1), dtype=targetSegsp4d.dtype)*3,targetSegsp4d))
nontargetSegsp4d = segments1d[targetSegments1d==False,:,5]
NTwithclassp41d = np.hstack((np.zeros((nontargetSegsp4d.shape[0], 1), dtype=nontargetSegsp4d.dtype)-2,nontargetSegsp4d))

classDatap41=np.vstack((Twithclassp41b,Twithclassp41p,Twithclassp41d,NTwithclassp41b,NTwithclassp41p,NTwithclassp41d))
classDatap41=np.hstack((np.ones((classDatap41.shape[0], 1)),classDatap41))
print classDatap41.shape,"$$$$$$$$$$$$DDDDWQQQ"

m = 210
I = np.eye(m)
D = np.diff(I,2).T
lamb = 500
DI = I + lamb * np.dot(D.T,D)

smoothed1b = np.zeros_like(P4data41b)
smoothed1p = np.zeros_like(P4data41p)
smoothed1d = np.zeros_like(P4data41d)
for trial in range(80):
    smoothed1b[trial,:] = np.linalg.solve(DI,P4data41b[trial,:])
    smoothed1p[trial,:] = np.linalg.solve(DI,P4data41p[trial,:])
    smoothed1d[trial,:] = np.linalg.solve(DI,P4data41d[trial,:])

targetSegs41bsmoothed = smoothed1b[targetSegments1b,:]
print targetSegs41bsmoothed.shape
Twithclass41bsmoothed = np.hstack((np.ones((targetSegs41bsmoothed.shape[0], 1), dtype=targetSegs41bsmoothed.dtype),targetSegs41bsmoothed))
nontargetSegs41bsmoothed = smoothed1b[targetSegments1b==False,:]
NTwithclass41bsmoothed = np.hstack((np.zeros((nontargetSegs41bsmoothed.shape[0], 1), dtype=nontargetSegs41bsmoothed.dtype),nontargetSegs41bsmoothed))
classData41bsmoothed=np.vstack((Twithclass41bsmoothed,NTwithclass41bsmoothed))

targetSegs41psmoothed = smoothed1p[targetSegments1p,:]
print targetSegs41psmoothed.shape
Twithclass41psmoothed = np.hstack((np.ones((targetSegs41psmoothed.shape[0], 1), dtype=targetSegs41psmoothed.dtype)*2,targetSegs41psmoothed))
nontargetSegs41psmoothed = smoothed1p[targetSegments1p==False,:]
NTwithclass41psmoothed = np.hstack((np.zeros((nontargetSegs41psmoothed.shape[0], 1), dtype=nontargetSegs41psmoothed.dtype)-1,nontargetSegs41psmoothed))
classData41psmoothed=np.vstack((Twithclass41psmoothed,NTwithclass41psmoothed))

targetSegs41dsmoothed = smoothed1d[targetSegments1d,:]
print targetSegs41dsmoothed.shape
Twithclass41dsmoothed = np.hstack((np.ones((targetSegs41dsmoothed.shape[0], 1), dtype=targetSegs41dsmoothed.dtype)*3,targetSegs41dsmoothed))
nontargetSegs41dsmoothed = smoothed1d[targetSegments1d==False,:]
NTwithclass41dsmoothed = np.hstack((np.zeros((nontargetSegs41dsmoothed.shape[0], 1), dtype=nontargetSegs41dsmoothed.dtype)-2,nontargetSegs41dsmoothed))
classData41dsmoothed=np.vstack((Twithclass41dsmoothed,NTwithclass41dsmoothed))

classDatap41smoothed=np.vstack((Twithclass41bsmoothed,Twithclass41psmoothed,Twithclass41dsmoothed,NTwithclass41bsmoothed,NTwithclass41psmoothed,NTwithclass41dsmoothed))
classData41smoothed=np.hstack((np.ones((classDatap41smoothed.shape[0], 1)),classDatap41smoothed))

np.savetxt('P4all3protS11smoothed.txt', classData41smoothed)
########################################SUBJECT 2#############################################

for i,(s,stim,targ) in enumerate(zip(starts2b,stimuli2b,targetSegments2b)):
   print s,stim,targ,'; ',
   if (i+1) % 5 == 0:
       print

indices2b = np.array([ np.arange(s,s+210) for s in starts2b ])
segments2b = eeg2b[indices2b,:8]

for i,(s,stim,targ) in enumerate(zip(starts2p,stimuli2p,targetSegments2p)):
   print s,stim,targ,'; ',
   if (i+1) % 5 == 0:
       print

indices2p = np.array([ np.arange(s,s+210) for s in starts2p ])
segments2p = eeg2p[indices2p,:8]

for i,(s,stim,targ) in enumerate(zip(starts2d,stimuli2d,targetSegments2d)):
   print s,stim,targ,'; ',
   if (i+1) % 5 == 0:
       print

indices2d = np.array([ np.arange(s,s+210) for s in starts2d ])
segments2d = eeg2b[indices2d,:8]

targetSegsp3b = segments2b[targetSegments2b,:,4]
Twithclassp32b = np.hstack((np.ones((targetSegsp3b.shape[0], 1), dtype=targetSegsp3b.dtype),targetSegsp3b))
nontargetSegsp3b = segments2b[targetSegments2b==False,:,4]
NTwithclassp32b = np.hstack((np.zeros((nontargetSegsp3b.shape[0], 1), dtype=nontargetSegsp3b.dtype),nontargetSegsp3b))

targetSegsp3p = segments2p[targetSegments2p,:,4]
Twithclassp32p = np.hstack((np.ones((targetSegsp3p.shape[0], 1), dtype=targetSegsp3p.dtype),targetSegsp3p))
nontargetSegsp3p = segments2p[targetSegments2p==False,:,4]
NTwithclassp32p = np.hstack((np.zeros((nontargetSegsp3p.shape[0], 1), dtype=nontargetSegsp3p.dtype),nontargetSegsp3p))

targetSegsp3d = segments2d[targetSegments2d,:,4]
Twithclassp32d = np.hstack((np.ones((targetSegsp3d.shape[0], 1), dtype=targetSegsp3d.dtype),targetSegsp3d))
nontargetSegsp3d = segments2d[targetSegments2d==False,:,4]
NTwithclassp32d = np.hstack((np.zeros((nontargetSegsp3d.shape[0], 1), dtype=nontargetSegsp3d.dtype),nontargetSegsp3d))

classDatap32=np.vstack((Twithclassp32b,Twithclassp32p,Twithclassp32d,NTwithclassp32b,NTwithclassp32p,NTwithclassp32d))
classDatap32=np.hstack((2*np.ones((classDatap32.shape[0], 1)),classDatap32))

targetSegsp4b = segments2b[targetSegments2b,:,5]
Twithclassp42b = np.hstack((np.ones((targetSegsp4b.shape[0], 1), dtype=targetSegsp4b.dtype),targetSegsp4b))
nontargetSegsp4b = segments2b[targetSegments2b==False,:,5]
NTwithclassp42b = np.hstack((np.zeros((nontargetSegsp4b.shape[0], 1), dtype=nontargetSegsp4b.dtype),nontargetSegsp4b))

targetSegsp4p = segments2p[targetSegments2p,:,5]
Twithclassp42p = np.hstack((np.ones((targetSegsp4p.shape[0], 1), dtype=targetSegsp4p.dtype),targetSegsp4p))
nontargetSegsp4p = segments2p[targetSegments2p==False,:,5]
NTwithclassp42p = np.hstack((np.zeros((nontargetSegsp4p.shape[0], 1), dtype=nontargetSegsp4p.dtype),nontargetSegsp4p))

targetSegsp4d = segments2d[targetSegments2d,:,5]
Twithclassp42d = np.hstack((np.ones((targetSegsp4d.shape[0], 1), dtype=targetSegsp4d.dtype),targetSegsp4d))
nontargetSegsp4d = segments2d[targetSegments2d==False,:,5]
NTwithclassp42d = np.hstack((np.zeros((nontargetSegsp4d.shape[0], 1), dtype=nontargetSegsp4d.dtype),nontargetSegsp4d))

classDatap42=np.vstack((Twithclassp42b,Twithclassp42p,Twithclassp42d,NTwithclassp42b,NTwithclassp42p,NTwithclassp42d))
classDatap42=np.hstack((2*np.ones((classDatap42.shape[0], 1)),classDatap42))


P3all2sub3prot = np.vstack((classDatap31,classDatap32))
P4all2sub3prot = np.vstack((classDatap41,classDatap42))
sci.savemat('P3all2sub3prot.mat', mdict = {'classData':P3all2sub3prot})
sci.savemat('P4all2sub3prot.mat', mdict = {'classData':P4all2sub3prot})

np.savetxt('P3all2sub3prot.txt', P3all2sub3prot)
np.savetxt('P4all2sub3prot.txt', P4all2sub3prot)
np.savetxt('P3all3protS11.txt', classDatap31)
np.savetxt('P3all3protS20.txt', classDatap32)
np.savetxt('P4all3protS11.txt', classDatap41)
np.savetxt('P4all3protS20.txt', classDatap42)
