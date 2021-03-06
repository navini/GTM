import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sci
from matplotlib.lines import Line2D
def summarize(datalist):
    for i,element in enumerate(datalist):
        keys = element.keys()
        print '\nData set', i
        keys.remove('eeg')
        for key in keys:
            print '  {}: {}'.format(key,element[key])
        eegtrials = element['eeg']
        shape = np.array(eegtrials['trial 1']).shape
        print ('  eeg: {:d} trials, each a matrix with {:d} rows' +
              ' and approximately {:d} columns').format( \
            len(eegtrials), shape[0], shape[1])


data = json.load(open('s11-gammasys-home-impaired.json','r'))
summarize(data)
letterd = data[6]
print letterd['protocol']


eeg = np.array(letterd['eeg']['trial 1']).T

print eeg.shape

plt.ion()

plt.figure(1);

plt.plot(eeg[4000:5000,:9] + 80*np.arange(8,-1,-1));

plt.plot(np.zeros((1000,9)) + 80*np.arange(8,-1,-1),'--',color='gray');

plt.yticks([]);

plt.legend(letterd['channels'] + ['Stimulus']);

plt.axis('tight');
plt.title('EEG Signals of 8 channels with stimulus (Single-letter D Protocol)');
plt.show()

print np.unique(eeg[:,8])
print [chr(int(n)) for n in np.abs(np.unique(eeg[:,8]))]


# Plot stimulus and P4
plt.figure(2);

plt.plot(eeg[3000:4500,[5,8]]);  #  Stimulus and P4

plt.axis('tight');
plt.title('EEG Signals of P4 channel with stimulus (Single-letter D Protocol)');
plt.xlabel('Time Samples');


#Collect all of the segments from the letter-d trial
starts = np.where(np.diff(np.abs(eeg[:,-1])) > 0)[0]
stimuli = [chr(int(n)) for n in np.abs(eeg[starts+1,-1])]
targetLetter = letterd['protocol'][-1]
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

targetSegs = segments[targetSegments,:,5]
Twithclass = np.hstack((np.ones((targetSegs.shape[0], 1), dtype=targetSegs.dtype),targetSegs))
nontargetSegs = segments[targetSegments==False,:,5]
NTwithclass = np.hstack((np.zeros((nontargetSegs.shape[0], 1), dtype=nontargetSegs.dtype),nontargetSegs))
print "P4, target segment shape", targetSegs.shape
print "P4, nontarget segment shape", nontargetSegs.shape
classData=np.vstack((Twithclass,NTwithclass))

np.savetxt('S11LDP4.txt', classData)

m = 210
I = np.eye(m)
D = np.diff(I,2).T
lamb = 500
DI = I + lamb * np.dot(D.T,D)
y = segments[0,:,5]
print y.shape
z = np.linalg.solve(DI,y)
plt.figure(3)
plt.plot(y)
plt.plot(z)
plt.show()

smoothed = np.zeros_like(segments[:,:,5])
for trial in range(80):
    smoothed[trial,:] = np.linalg.solve(DI,segments[trial,:,5])
'''
stimulus = eeg[:,-1]
letter = np.array([chr(int(n)) for n in abs(stimulus[starts+1])])

plt.figure(figsize=(20,20))
for trial in range(80):
    plt.subplot(10,8,trial+1)
    plt.plot(segments[trial,:,4])
    if letter[trial] == 'b':
        color = 'r'
    else:
        color = 'b'
    plt.plot(-20+smoothed[trial,:],color)'''

print smoothed.shape

targetSegs4smoothed = smoothed[targetSegments,:]
print targetSegs4smoothed.shape
Twithclass4smoothed = np.hstack((np.ones((targetSegs4smoothed.shape[0], 1), dtype=targetSegs4smoothed.dtype),targetSegs4smoothed))
nontargetSegs4smoothed = smoothed[targetSegments==False,:]
NTwithclass4smoothed = np.hstack((np.zeros((nontargetSegs4smoothed.shape[0], 1), dtype=nontargetSegs4smoothed.dtype),nontargetSegs4smoothed))
classData4smoothed=np.vstack((Twithclass4smoothed,NTwithclass4smoothed))

print classData4smoothed.shape

np.savetxt('S11LDP4-smoothed.txt', classData4smoothed)

print segments[targetSegments,:,5].shape
markers = np.array(['b--','b--','k--','k--','r-','r--','c--','c--','g-','g--','m--','y--','m--','y-','c:','k:','b:','r:','g:','m-','y-','m_'])
plt.figure(4)
yplot = (segments[targetSegments,:,5]).T
for i in range(20):
    plt.plot( (np.arange(210) / 256.0 * 1000),yplot[:,i],markers[i])
plt.xlim(0, 210);
plt.axis('tight');

plt.figure(5);
aver = np.array([1,2,5,10,15,20])
xs = np.arange(210) / 256.0 * 1000  # milliseconds

targets = segments[targetSegments,:,5]
print targets.shape,'@@@'
ntargets = segments[targetSegments==False,:,5]
for i in range(6):
    plt.subplot(3,2,i+1)
    targetMean = np.mean(targets[0:aver[i],:],axis=0)
    print targetMean
    print '_________________________________'
    nontargetMean = np.mean(ntargets[0:aver[i],:],axis=0)
    plt.plot(xs, np.vstack((targetMean,nontargetMean)).T);
    plt.legend(('Target','Nontarget'));
    plt.axis('tight');
    plt.axhline(y=0, color='k',linestyle='dashed')
    #plt.axvline(x=(101/256.0*1000), ymin=0, color='k',linestyle='dashed')
    plt.xlabel('Milliseconds After Stimulus');
    plt.ylabel('Mean Voltage ($\mu$ V)');
    plt.title(aver[i])


print segments[targetSegments,:,5].shape
#targetMean = np.mean(segments[targetSegments,:,5],axis=0)
targetMean = np.mean(targets[0:20,:],axis=0)
print targetMean
nontargetMean = np.mean(segments[targetSegments==False,:,5],axis=0)
plt.figure(6);
#xs = np.arange(len(targetMean)) / 256.0 * 1000  # milliseconds
plt.plot(xs, np.vstack((targetMean,nontargetMean)).T);
plt.legend(('Target','Nontarget'));
plt.axis('tight');
plt.axhline(y=0, color='k',linestyle='dashed')
plt.axvline(x=(101/256.0*1000), ymin=0, color='k',linestyle='dashed')
plt.xlabel('Milliseconds After Stimulus');
plt.ylabel('Mean Voltage ($\mu$ V)');
