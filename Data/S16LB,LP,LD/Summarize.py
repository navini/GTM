import numpy as np
import json as js
import matplotlib.pyplot as plt


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


data = js.load(open('s20-gammasys-gifford-unimpaired.json','r'))
print(len(data))     #8  - 8 Datasets
print("\n")
print(data[0].keys())   
'''[u'protocol',
 u'sample rate',
 u'notes',
 u'channels',
 u'date',
 u'location',
 u'device',
 u'eeg',
 u'impairment',
 u'subject']  '''


print("\n")
summarize(data)
print("\n")

first = data[0]
eeg = np.array(first['eeg']['trial 1'])   #Data set 0
print(eeg.shape)   #[9,46330] 9=eeg channels ['F3','F4','C3','C4','P3','P4','O1','O2']+stimulus onset/offset channel 46330=256 samples per sec * 3 mins

# Plot all channels for column 4000-4512
plt.figure(1);
plt.plot(eeg[:,4000:4512].T);
plt.axis('tight');
plt.show()


#Ignore 9th channel, add legend
plt.figure(2);
plt.plot(eeg[:8,4000:4512].T + 80*np.arange(7,-1,-1));
plt.plot(np.zeros((512,8)) + 80*np.arange(7,-1,-1),'--',color='gray');
plt.yticks([]);
plt.legend(first['channels']);
plt.axis('tight');
plt.show()
