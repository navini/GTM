# neuralnetworks.py:  multiple hidden layers, and code consolidation

import ScaledConjugateGradient as SCG
# reload(SCG)
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp  # to allow access to number of elapsed iterations
import cPickle
import mlutils as ml
reload(ml)

def pickleLoad(filename):
    with open(filename,'rb') as fp:
        nnet = cPickle.load(fp)
    nnet.iteration = mp.Value('i',0)
    nnet.trained = mp.Value('b',False)
    return nnet

class NeuralNetwork:
    """ Neural network with multiple hidden layers.
 For nonlinear regression (prediction of real-valued outputs)
   net = NeuralNetwork(ni,nh,no)       # ni is number of attributes each sample,
                                   # nh is number of hidden units, possibly a tuple or list
                                   # no is number of output components
   net.train(X,T,                  # X is nSamples x ni, T is nSamples x no
             nIterations=1000,     # maximum number of SCG iterations
             weightPrecision=1e-8, # SCG terminates when weight change magnitudes fall below this,
             errorPrecision=1e-8)  # SCG terminates when objective function change magnitudes fall below this
   Y,Z = net.use(Xtest,allOutputs=True)  # Y is nSamples x no, Z is nSamples x nh

 For nonlinear classification (prediction of integer valued class labels)
   net = NeuralNetworkClassifier(ni,nh,no)
   net.train(X,T,                  # X is nSamples x ni, T is nSamples x 1 (integer class labels
             nIterations=1000,     # maximum number of SCG iterations
             weightPrecision=1e-8, # SCG terminates when weight change magnitudes fall below this,
             errorPrecision=1e-8)  # SCG terminates when objective function change magnitudes fall below this
   classes,Y,Z = net.use(Xtest,allOutputs=True)  # classes is nSamples x 1
"""
    def __init__(self,ni,nhs,no):

        # if nhs is scalar, make it a list of one element
        try:
            l = len(nhs)
        except:
            nhs = (nhs,)

        nihs = [ni] + list(nhs)

        # Initialize weights
        self.Vs = [np.random.uniform(-0.1,0.1,size=(1+nihs[i],nihs[i+1])) for i in range(len(nihs)-1)]
        self.W = np.random.uniform(-0.1,0.1,size=(1+nhs[-1],no))
        # print [v.shape for v in self.Vs], self.W.shape

        self.ni,self.nhs,self.no = ni,nhs,no

        # Initialize standardization parameters
        self.Xmeans = None
        self.Xstds = None
        self.Tmeans = None
        self.Tstds = None

        self.iteration = mp.Value('i',0)
        self.trained = mp.Value('b',False)
        
    def pickleDump(self,filename):
        """Save neural network to filename"""
        # remove shared memory objects. Can't be pickled
        n = self.iteration.value
        t = self.trained.value
        self.iteration = None
        self.trained = None
        with open(filename,'wb') as fp:
        #    pickle.dump(self,fp)
            cPickle.dump(self,fp)
        self.iteration = mp.Value('i',n)
        self.trained = mp.Value('b',t)
            
    def getSize(self):
        """Return number of inputs, list of numbers of hidden units per layer, and number of outputs"""
        return (self.ni,self.nhs,self.no)
    
    def train(self,X,T, nIterations=100,weightPrecision=1e-10,errorPrecision=1e-10):
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
        X = self.standardizeX(X)

        if T.ndim == 1:
            T = T.reshape((-1,1))

        if self.Tmeans is None:
            self.Tmeans = T.mean(axis=0)
            self.Tstds = T.std(axis=0)
        T = self.standardizeT(T)

        scgresult = SCG.scg(self._pack(self.Vs,self.W),
                            self.objectiveF, self.gradF, X,T,
                            xPrecision = weightPrecision,
                            fPrecision = errorPrecision,
                            nIterations = nIterations,
                            iterationVariable = self.iteration,
                            ftracep=True)

        self._unpack(scgresult['x'])
        self.reason = scgresult['reason']
        self.errorTrace = scgresult['ftrace']
        self.numberOfIterations = len(self.errorTrace)
        self.trained.value = True
        return self
    
    def objectiveF(self,w,X,T):
        self._unpack(w)
        Y,_ = self._forwardPass(X)
        return 0.5 * np.mean((Y - T)**2)

    def gradF(self,w,X,T):
        self._unpack(w)
        Y,Zs = self._forwardPass(X)
        delta = (Y - T) / (X.shape[0] * T.shape[1])
        dVs,dW = self._backwardPass(delta,Zs)
        return self._pack(dVs,dW)

    # Local functions for use with ScaledConjugateGradient.py
    def _pack(self,Vs,W):
        return np.hstack([V.flat for V in Vs] + [W.flat])
    def _unpack(self,w):
            first = 0
            numInThisLayer = self.ni
            for i in range(len(self.Vs)):
                self.Vs[i][:] = w[first:first+(numInThisLayer+1)*self.nhs[i]].reshape((numInThisLayer+1,self.nhs[i]))
                first += (numInThisLayer+1) * self.nhs[i]
                numInThisLayer = self.nhs[i]
            self.W[:] = w[first:].reshape((numInThisLayer+1,self.no))
    def _forwardPass(self,X):
        Zprev = X
        Zs = [Zprev]
        for i in range(len(self.nhs)):
            V = self.Vs[i]
            Zprev = np.tanh(np.dot(Zprev,V[1:,:]) + V[0:1,:])
            Zs.append(Zprev)
        Y = np.dot(Zprev, self.W[1:,:]) + self.W[0:1,:]
        return Y,Zs

    def _backwardPass(self,delta,Zs):
        dW = np.vstack((np.dot(np.ones((1,delta.shape[0])),delta),  np.dot( Zs[-1].T, delta)))
        dVs = []
        delta = (1-Zs[-1]**2) * np.dot( delta, self.W[1:,:].T)
        for Zi in range(len(self.nhs),0,-1):
            Vi = Zi - 1 # because X is first element of Z
            dV = np.vstack(( np.dot(np.ones((1,delta.shape[0])), delta),
                             np.dot( Zs[Zi-1].T, delta)))
            dVs.insert(0,dV)
            delta = np.dot( delta, self.Vs[Vi][1:,:].T) * (1-Zs[Zi-1]**2)
        return dVs,dW

    def getNumberOfIterations(self):
        return self.numberOfIterations
        
    def use(self,X,allOutputs=False):
        X = self.standardizeX(X)
        Y,Zs = self._forwardPass(X)
        if self.Tmeans:
            Y = self.unstandardizeT(Y)
        return (Y,Zs[1:]) if allOutputs else Y

    def getErrorTrace(self):
        return self.errorTrace
    
    def plotErrors(self):
        plt.plot(self.errorTrace)
        plt.ylabel("Train RMSE")
        plt.xlabel("Iteration")

    def standardizeX(self,X):
        return (X - self.Xmeans) / self.Xstds
    def unstandardizeX(self,Xs):
        return self.Xstds * Xs + self.Xmeans
    def standardizeT(self,T):
        return (T - self.Tmeans) / self.Tstds
    def unstandardizeT(self,Ts):
        return self.Tstds * Ts + self.Tmeans

class NeuralNetworkClassifier(NeuralNetwork):

    def __init__(self,ni,nhs,no):
        NeuralNetwork.__init__(self,ni,nhs,no-1)

    def train(self,X,T, nIterations=100,weightPrecision=1e-10,errorPrecision=1e-10):
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
        X = self.standardizeX(X)

        self.classes = np.unique(T)
        if self.no != len(self.classes)-1:
            raise ValueError(" In NeuralNetworkClassifier, the number of outputs must be one less than\n the number of classes in the training data. The given number of outputs\n is %d and number of classes is %d. Try changing the number of outputs in the\n call to NeuralNetworkClassifier()." % (self.no, len(self.classes)))
        T = ml.makeIndicatorVars(T)

        scgresult = SCG.scg(self._pack(self.Vs,self.W), self.objectiveF, self.gradF, X,T,
                           xPrecision = weightPrecision,
                           fPrecision = errorPrecision,
                           nIterations = nIterations,
                           iterationVariable = self.iteration,
                           ftracep=True)

        self._unpack(scgresult['x'])
        self.reason = scgresult['reason']
        self.errorTrace = scgresult['ftrace']
        self.numberOfIterations = len(self.errorTrace)
        self.trained.value = True
        return self

    def objectiveF(self,w,X,T):
        self._unpack(w)
        Y,_ = self._forwardPass(X)
        expY = np.exp(Y)
        denom = 1 + np.sum(expY,axis=1).reshape((-1,1))
        Y = np.hstack((expY / denom, 1/denom))
        return -np.mean(T * np.log(Y))

    def gradF(self,w,X,T):
        self._unpack(w)
        Y,Zs = self._forwardPass(X)
        expY = np.exp(Y)
        denom = 1 + np.sum(expY,axis=1).reshape((-1,1))
        Y = np.hstack((expY /denom , 1.0/denom))
        delta = (Y[:,:-1] - T[:,:-1]) / (X.shape[0] * (T.shape[1]-1))
        dVs,dW = self._backwardPass(delta,Zs)
        return self._pack(dVs,dW)

    def use(self,X,allOutputs=False):
        X = self.standardizeX(X)
        Y,Zs = self._forwardPass(X)
        expY = np.exp(Y)
        denom = 1 + np.sum(expY,axis=1).reshape((-1,1))
        Y = np.hstack((expY / denom, 1/denom))
        classes = self.classes[np.argmax(Y,axis=1)].reshape((-1,1))
        return (classes,Y,Zs) if allOutputs else classes
        Y = np.dot(Zprev, self.W[1:,:]) + self.W[0:1,:]

    def plotErrors(self):
        plt.plot(np.exp(-self.errorTrace))
        plt.ylabel("Train Likelihood")
        plt.xlabel("Iteration")

class NeuralNetworkQ(NeuralNetwork):

    def __init__(self,ni,nh,no,inputminmax):
        NeuralNetwork.__init__(self,ni,nh,no)
        inputminmaxnp = np.array(inputminmax)
        self.Xmeans = inputminmaxnp.mean(axis=0)
        self.Xstds = inputminmaxnp.std(axis=0)

    def train(self,X,R,Q,Y,gamma=1, nIterations=100,weightPrecision=1e-10,errorPrecision=1e-10):
        X = self.standardizeX(X)
        T = R + gamma * Q
        
        scgresult = SCG.scg(self._pack(self.Vs,self.W), self.objectiveF, self.gradF, X,T, 
                            xPrecision = weightPrecision,
                            fPrecision = errorPrecision,
                            nIterations = nIterations,
                            iterationVariable = self.iteration,
                            ftracep=True)

        self._unpack(scgresult['x'])
        self.reason = scgresult['reason']
        self.errorTrace = scgresult['ftrace']
        self.numberOfIterations = len(self.errorTrace)
        self.trained.value = True
        return self

if __name__== "__main__":
    plt.ion()

    print '\n------------------------------------------------------------'
    print "Regression Example: Approximate f(x) = 1.5 + 0.6 x + 0.4 sin(x)"
    # print '                    Neural net with 1 input, 5 hidden units, 1 output'
    nSamples = 20
    X = np.linspace(0,10,nSamples).reshape((-1,1))
    T = 1.5 + 0.6 * X + 0.8 * np.sin(1.5*X)
    startBump = nSamples/5
    T[startBump:startBump+nSamples/10] *= 3
    startBump += 2*nSamples/10
    T[startBump:startBump+nSamples/10] *= 3
    
    nSamples = 50
    Xtest = np.linspace(0,9,nSamples).reshape((-1,1))
    Ttest = 1.5 + 0.6 * Xtest + 0.8 * np.sin(1.5*Xtest) + np.random.uniform(-1,1,size=(nSamples,1))
    startBump = nSamples/5
    Ttest[startBump:startBump+nSamples/10] *= 3
    startBump += 2*nSamples/10
    Ttest[startBump:startBump+nSamples/10] *= 3

    # # nnet = NeuralNetwork(1,(5,5,5,5,5,5,5,,5,),1)
    # # nnet = NeuralNetwork(1,(5,),1)
    # # nnet = NeuralNetwork(1,(5,4,3,2),1)
    # # nnet = NeuralNetwork(1,(5,5,5,5,5,2),1)
    # # nnet = NeuralNetwork(1,(10,2,10),1)
    # # nnet = NeuralNetwork(1,(10,1,10),1)
    # # nnet = NeuralNetwork(1,(5,5,),1)
    nnet = NeuralNetwork(1,10,1)
    
    nnet.train(X,T,weightPrecision=1.e-20,errorPrecision=1.e-20,nIterations=10000)
    print "SCG stopped after",nnet.getNumberOfIterations(),"iterations:",nnet.reason
    (Y,Z) = nnet.use(X, allOutputs=True)
    Ytest = nnet.use(Xtest)
    print "Final RMSE: train", np.sqrt(np.mean((Y-T)**2)),"test",np.sqrt(np.mean((Ytest-Ttest)**2))
    
    # print 'Inputs, Targets, Estimated Targets'
    # print np.hstack((X,T,Y))

    plt.figure(1)
    plt.clf()
    
    nPlotRows = 4 + len(nnet.nhs)

    plt.subplot(nPlotRows,1,1)
    nnet.plotErrors()
    plt.title('Regression Example')
    plt.subplot(nPlotRows,1,2)
    plt.plot(X,T)
    plt.plot(X,Y)
    plt.legend(('Train Target','Train NN Output'),loc='lower right',
               prop={'size':9})
    plt.subplot(nPlotRows,1,3)
    plt.plot(Xtest,Ttest)
    plt.plot(Xtest,Ytest)
    plt.legend(('Test Target','Test NN Output'),loc='lower right',
               prop={'size':9})
    colors = ('blue','green','red','black','cyan','orange')
    for i in range(len(nnet.nhs)):
        plt.subplot(nPlotRows,1,i+4)
        plt.plot(X,Z[i]) #,color=colors[i])
        plt.ylabel('Hidden Unit Outputs')
        plt.text(8,0, 'Layer {}'.format(i+1))

    plt.subplot(nPlotRows,1,nPlotRows)
    ml.draw(nnet.Vs,nnet.W,['x'],['sine'])
    plt.draw()
    
    # Now train multiple nets to compare error for different numbers of hidden layers

    def experiment(hs,nReps,nIter,X,T,Xtest,Ytest):
        results = []
        for i in range(nReps):
            nnet = NeuralNetwork(1,hs,1)
            nnet.train(X,T,weightPrecision=0,errorPrecision=0,nIterations=nIter)
            # print "SCG stopped after",nnet.getNumberOfIterations(),"iterations:",nnet.reason
            (Y,Z) = nnet.use(X, allOutputs=True)
            Ytest = nnet.use(Xtest)
            rmsetrain = np.sqrt(np.mean((Y-T)**2))
            rmsetest = np.sqrt(np.mean((Ytest-Ttest)**2))
            results.append([rmsetrain,rmsetest])
        return results

    plt.figure(2)
    plt.clf()
    
    results = []
    # hiddens [ [5]*i for i in range(1,6) ]
    hiddens = [[12], [6,6], [4,4,4], [3,3,3,3], [2,2,2,2,2,2],
               [24], [12]*2, [8]*3, [6]*4, [4]*6, [3]*8, [2]*12]
    for hs in hiddens:
        r = experiment(hs,3,100,X,T,Xtest,Ttest)
        r = np.array(r)
        means = np.mean(r,axis=0)
        stds = np.std(r,axis=0)
        results.append([hs,means,stds])
        print hs, means,stds
        rmseMeans = np.array([x[1].tolist() for x in results])
        plt.clf()
        plt.plot(rmseMeans,'o-')
        ax = plt.gca()
        plt.xticks(range(len(hiddens)),[str(h) for h in hiddens])
        plt.draw()
        


    print '\n------------------------------------------------------------'
    print "Classification Example: XOR, approximate f(x1,x2) = x1 xor x2"
    print '                        Using neural net with 2 inputs, 3 hidden units, 2 outputs'
    X = np.array([[0,0],[1,0],[0,1],[1,1]])
    T = np.array([[1],[2],[2],[1]])
    nnet = NeuralNetworkClassifier(2,(3,3,3),2)
    nnet.train(X,T,weightPrecision=1.e-8,errorPrecision=1.e-8,nIterations=100)
    print "SCG stopped after",nnet.getNumberOfIterations(),"iterations:",nnet.reason
    (classes,y,Z) = nnet.use(X, allOutputs=True)
    

    print 'X(x1,x2), Target Classses, Predicted Classes'
    print np.hstack((X,T,classes))

    print "Hidden Outputs"
    print Z
    
    plt.figure(3)
    plt.clf()
    plt.subplot(2,1,1)
    nnet.plotErrors()
    plt.title('Classification Example')
    plt.subplot(2,1,2)
    ml.draw(nnet.Vs,nnet.W,['x1','x2'],['xor'])

    print '\n------------------------------------------------------------'
    print "Reinforcement Learning Example: "

    import random

    def epsilonGreedy(net,s,epsilon):
        qs = net.use(np.array([[s,a] for a in [-1,0,1]]))
        if np.random.uniform() < epsilon:
            # random action
            a = random.randint(-1,1)
            i = a+1
            Q = qs[i,0]
        else:
            i = np.argmax(qs)
            a = [-1,0,1][i]
            Q = qs[i,0]
        return (a,Q)

    def getSamples(net,N,epsilon):
        X = np.zeros((N,2))
        R = np.zeros((N,1))
        Q = np.zeros((N,1))
        Y = np.zeros((N,1))
        # Initial state  1 to 3
        s = random.randint(1,3)  
        # Select action using epsilon-greedy policy and get Q
        a,q = epsilonGreedy(net,s,epsilon)
        for step in xrange(N):
            # Update state, s1 from s and a
            s1 = min(3,max(1, s + a))
            # Get resulting reinforcement = 1 if state = 2, 0 otherwise
            r1 = 1 if s1 == 2 else 0
            # Select action for next step and get Q
            a1,q1 = epsilonGreedy(net,s1,epsilon)
            # Collect
            X[step,:], R[step,0], Q[step,0], Y[step,0] = (s,a),r1,q1,q
            # Shift state, action and action index by one time step
            s,a,q = s1,a1,q1
        return X,R,Q,Y

    N = 100
    epsilon = 1
    nnet = NeuralNetworkQ(2,5,1,((1,3),(-1,1)))
    Rtrace = []
    Xtrace = []
    for reps in range(10):
        print 'exploration probability, epsilon, is',epsilon
        X,R,Q,Y = getSamples(nnet,N,epsilon)
        nnet.train(X,R,Q,Y,gamma=0.9,nIterations=200,
                  weightPrecision=0,errorPrecision=0)
        epsilon *= 0.5
        Rtrace += R.ravel().tolist()
        Xtrace += X[:,0].ravel().tolist()

    qs = nnet.use(np.array([[s,a] for a in [-1,0,1] for s in [1,2,3]]))

    qsstr = [q[0] for q in qs]
    print 'Three states, actions are left, stay, right. Walls at ends. Goal is middle.'
    print 'Highest Q value on each row in order should be in:'
    print '  third column (move right),'
    print '  middle column (stay put),'
    print '  first column (move left),'
    print
    print '          {:10s} {:10s} {:10s}\n  1 {:10.2f} {:10.2f} {:10.2f}\n  2 {:10.2f} {:10.2f} {:10.2f}\n  3 {:10.2f} {:10.2f} {:10.2f}'.format('Left','Stay','Right',*qsstr)
    
    plt.figure(3)
    plt.subplot(2,2,1)
    plt.plot(Rtrace)
    nsteps = len(Rtrace)
    npoints = 100
    window = int(nsteps/npoints)
    xnew = np.arange(0,nsteps,window)
    ym = np.mean(np.array(Rtrace[:npoints*window]).reshape(-1,window), axis=1)
    plt.plot(xnew+window/2, ym,lw=3,alpha=0.5,c='red')
    plt.ylim(-0.1,1.1)
    plt.title('Reinforcement Learning Example')
    plt.ylabel('Reward')
    plt.xlabel('Step')
    plt.subplot(2,2,2)
    ml.draw(nnet.Vs,nnet.W,['s','a'],['Q'])
    plt.subplot(2,2,3)
    plt.plot(Xtrace)
    plt.ylim(0.9,3.1)
    plt.ylabel('State')
    plt.xlabel('Step')
    plt.show()
