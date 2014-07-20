from __future__ import division
from matplotlib import pyplot as plt
import numpy as np
import neuralnetworks as nn
import sys

def normalize(signal):
    means = np.mean(signal, axis=1).reshape((signal.shape[0],1))
    std = np.std(signal, axis=1).reshape((signal.shape[0],1))
    signal = np.divide(np.subtract(signal,means),std)
    return signal

data=np.loadtxt(sys.argv[1])
BSR = np.zeros((20,1))

for j in range(10):
    for k in range(20):
        data_features = data[:,1:211]
        data_classes = data[:,0:1]

        TP=0
        TN=0
        FP=0
        FN=0

        for i in range(data_features.shape[0]):
            print i
            test_data = data_features[i:i+1,:]
            test_class = data_classes[i:i+1,:]
            train_data = np.delete(data_features, (i), axis=0 )
            train_classes = np.delete(data_classes, (i), axis=0 )

            nnet = nn.NeuralNetClassifier(210,k+1,2) # 2 inputs, 8 hiddens, 3 outputs
            nnet.train(train_data,train_classes,weightPrecision=1.e-8,errorPrecision=1.e-8,nIterations=400)
            print "SCG stopped after",nnet.getNumberOfIterations(),"iterations:",nnet.reason
            (classes,Y,Z) = nnet.use(data_features, allOutputs=True)


            marker = ['r.','bx']

            predTest,probs,Z = nnet.use(test_data,allOutputs=True) 
            print test_class,' ',predTest



            ###########Print Confusion Matrix



            if(predTest[0,:]==test_class[0,:]):
                if(predTest[0,:]==0):
                    TN=TN+1
                else:
                    TP=TP+1
            else:
                if(predTest[0,:]==0):
                    FN=FN+1
                else:
                    FP=FP+1

        print '\t Confusion matrix'

        print '\t    predicted labels'
        print '\t \t 0 \t 1'

        print 'true lables 0   ',TN,'\t ',FP
        print '\t    1   ',FN,'\t ',TP

        BSR[k,0] = BSR[k,0]+((TP/(TP+FN) + TN/(TN+FP))/2)
print BSR*10
