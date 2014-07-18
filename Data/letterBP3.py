import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sci


data = json.load(open('s11-gammasys-home-impaired.json','r'))
letterb = data[2]
print letterb['protocol']


eeg = np.array(letterb['eeg']['trial 1']).T

print eeg.shape

plt.ion()

plt.figure();

plt.plot(eeg[1000:5000,:9] + 80*np.arange(8,-1,-1));

plt.plot(np.zeros((1000,9)) + 80*np.arange(8,-1,-1),'--',color='gray');

plt.yticks([]);

plt.legend(letterb['channels'] + ['Stimulus']);

plt.axis('tight');

print np.unique(eeg[:,8])
print [chr(int(n)) for n in np.abs(np.unique(eeg[:,8]))]


# Plot stimulus and P3
plt.clf();

plt.plot(eeg[3000:4500,[5,8]]);  #  Stimulus and P3

plt.axis('tight');

plt.xlabel('Time Samples');


#Collect all of the segments from the letter-b trial
starts = np.where(np.diff(np.abs(eeg[:,-1])) > 0)[0]
stimuli = [chr(int(n)) for n in np.abs(eeg[starts+1,-1])]
targetLetter = letterb['protocol'][-1]
targetSegments = np.array(stimuli) == targetLetter

print len(starts),len(stimuli),len(targetSegments)


for i,(s,stim,targ) in enumerate(zip(starts,stimuli,targetSegments)):
   print s,stim,targ,'; ',
   if (i+1) % 5 == 0:
       print

print "window lengths ",np.diff(starts)
print "minimum length ",np.min(np.diff(starts))

indices = np.array([ np.arange(s,s+210) for s in starts ])
print "indices shape ",indices.shape

segments = eeg[indices,:8]
print "sigment shape ",segments.shape

targetSegsp3 = segments[targetSegments,:,4]
Twithclassp3 = np.hstack((np.ones((targetSegsp3.shape[0], 1), dtype=targetSegsp3.dtype),targetSegsp3))
nontargetSegsp3 = segments[targetSegments==False,:,4]
NTwithclassp3 = np.hstack((np.zeros((nontargetSegsp3.shape[0], 1), dtype=nontargetSegsp3.dtype),nontargetSegsp3))
print "P4, target segment shape", targetSegsp3.shape
print "P4, nontarget segment shape", nontargetSegsp3.shape
classDatap3=np.vstack((Twithclassp3,NTwithclassp3))

targetSegsp4 = segments[targetSegments,:,5]
Twithclassp4 = np.hstack((np.ones((targetSegsp4.shape[0], 1), dtype=targetSegsp4.dtype),targetSegsp4))
nontargetSegsp4 = segments[targetSegments==False,:,5]
NTwithclassp4 = np.hstack((np.zeros((nontargetSegsp4.shape[0], 1), dtype=nontargetSegsp4.dtype),nontargetSegsp4))
print "P4, target segment shape", targetSegsp3.shape
print "P4, nontarget segment shape", nontargetSegsp3.shape
classDatap4=np.vstack((Twithclassp4,NTwithclassp4))

P3data = segments[:,:,4]
#print P3data
P4data = segments[:,:,5]
#print "P4, segment shape", P3data.shape

'''
f = open("data","w")
for i in range(P4data.shape[0]):
   row = P4data[i,:]
   print "row   ",row
   f.write(str(row))

f.close()'''
'''
sci.savemat('DataP3.mat', mdict = {'data':P3data})
sci.savemat('targetDataP3.mat', mdict = {'tagetSegs':targetSegs})
sci.savemat('nontargetDataP3.mat', mdict = {'nontagetSegs':nontargetSegs})
sci.savemat('target_classP3.mat', mdict = {'Tclass':Twithclass})
sci.savemat('nontarget_classP3.mat', mdict = {'NTclass':NTwithclass})'''
sci.savemat('classDataP3s11GB.mat', mdict = {'classData':classDatap3})
sci.savemat('classDataP4s11GB.mat', mdict = {'classData':classDatap4})
#np.savetxt('classDataP3.txt', classData)
'''
print Twithclass
print nontargetSegs
print NTwithclass

print "Full data", classData.shape
print classData


indices = np.array([ np.arange(s,s+4) for s in starts ])
print "indices shape ",indices.shape
segments = eeg[indices,:8]
print "sigment shape ",segments.shape

targetSegs4 = segments[targetSegments,:,5]
Twithclass4 = np.hstack((np.ones((targetSegs4.shape[0], 1), dtype=targetSegs4.dtype),targetSegs4))
nontargetSegs4 = segments[targetSegments==False,:,5]
NTwithclass4 = np.hstack((np.zeros((nontargetSegs4.shape[0], 1), dtype=nontargetSegs4.dtype),nontargetSegs4))
classData4=np.vstack((Twithclass4,NTwithclass4))

P4data4 = segments[:,:,5]
print P4data

sci.savemat('Data4.mat', mdict = {'data':P4data4})
sci.savemat('targetData4.mat', mdict = {'tagetSegs':targetSegs4})
sci.savemat('nontargetData4.mat', mdict = {'nontagetSegs':nontargetSegs4})
sci.savemat('target_class4.mat', mdict = {'Tclass':Twithclass4})
sci.savemat('nontarget_class4.mat', mdict = {'NTclass':NTwithclass4})
sci.savemat('classData4.mat', mdict = {'classData':classData4})
'''

