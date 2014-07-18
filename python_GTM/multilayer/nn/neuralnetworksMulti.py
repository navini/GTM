# neuralnetworks.py

import ScaledConjugateGradient as SCG
# reload(gd)
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp  # to allow access to number of elapsed iterations
import cPickle

def pickleLoad(filename):
    with open(filename,'rb') as fp:
        nnet = cPickle.load(fp)
    nnet.iteration = mp.Value('i',0)
    nnet.trained = mp.Value('b',False)
    return nnet

class NeuralNet:
    """ Neural network with one hidden layer.
 For nonlinear regression (prediction of real-valued outputs)
   net = NeuralNet(ni,nh,no)       # ni is number of attributes each sample,
                                   # nh is number of hidden units,
                                   # no is number of output components
   net.train(X,T,                  # X is nSamples x ni, T is nSamples x no
             nIterations=1000,     # maximum number of SCG iterations
             weightPrecision=1e-8, # SCG terminates when weight change magnitudes fall below this,
             errorPrecision=1e-8)  # SCG terminates when objective function change magnitudes fall below this
   Y,Z = net.use(Xtest,allOutputs=True)  # Y is nSamples x no, Z is nSamples x nh

 For nonlinear classification (prediction of integer valued class labels)
   net = NeuralNetClassifier(ni,nh,no)
   net.train(X,T,                  # X is nSamples x ni, T is nSamples x 1 (integer class labels
             nIterations=1000,     # maximum number of SCG iterations
             weightPrecision=1e-8, # SCG terminates when weight change magnitudes fall below this,
             errorPrecision=1e-8)  # SCG terminates when objective function change magnitudes fall below this
   classes,Y,Z = net.use(Xtest,allOutputs=True)  # classes is nSamples x 1
"""
    def __init__(self,ni,nhs,no):
        nihs = [ni] + list(nhs)
        self.Vs = [np.random.uniform(-0.1,0.1,size=(1+nihs[i],nihs[i+1])) for i in range(len(nihs)-1)]
        self.W = np.random.uniform(-0.1,0.1,size=(1+nhs[-1],no))
        # print [v.shape for v in self.Vs], self.W.shape
        self.ni,self.nhs,self.no = ni,nhs,no
        self.Xmeans = None
        self.Xstds = None
        self.Tmeans = None
        self.Tstds = None
        self.iteration = mp.Value('i',0)
        self.trained = mp.Value('b',False)
        
    def pickleDump(self,filename):
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
        return (self.ni,self.nhs,self.no)
    
    def train(self,X,T,
              nIterations=100,weightPrecision=0.0001,errorPrecision=0.0001):
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
        X = self.standardizeX(X)
        X1 = self.addOnes(X)

        if T.ndim == 1:
            T = T.reshape((-1,1))

        if self.Tmeans is None:
            self.Tmeans = T.mean(axis=0)
            self.Tstds = T.std(axis=0)
        T = self.standardizeT(T)

        # Local functions used by gradientDescent.scg()
        def pack(Vs,W):
            # r = np.hstack([V.flat for V in Vs] + [W.flat])
            # print 'pack',len(Vs), Vs[0].shape, W.shape,r.shape
            return np.hstack([V.flat for V in Vs] + [W.flat])
        def unpack(w):
            first = 0
            numInThisLayer = self.ni
            for i in range(len(self.Vs)):
                self.Vs[i][:] = w[first:first+(numInThisLayer+1)*self.nhs[i]].reshape((numInThisLayer+1,self.nhs[i]))
                first += (numInThisLayer+1) * self.nhs[i]
                numInThisLayer = self.nhs[i]
            self.W[:] = w[first:].reshape((numInThisLayer+1,self.no))
        def objectiveF(w):
            unpack(w)
            Zprev = X
            for i in range(len(self.nhs)):
                V = self.Vs[i]
                Zprev = np.tanh(np.dot(Zprev,V[1:,:]) + V[0:1,:])
            Y = np.dot(Zprev, self.W[1:,:]) + self.W[0:1,:]
            return 0.5 * np.mean((Y - T)**2)
        def gradF(w):
            unpack(w)
            Zprev = X
            Z = [Zprev]
            for i in range(len(self.nhs)):
                V = self.Vs[i]
                Zprev = np.tanh(np.dot(Zprev,V[1:,:]) + V[0:1,:])
                Z.append(Zprev)
            Y = np.dot(Zprev, self.W[1:,:]) + self.W[0:1,:]
            delta = (Y - T) / (X1.shape[0] * T.shape[1])
            dW = np.vstack((np.dot(np.ones((1,delta.shape[0])),delta),  np.dot( Z[-1].T, delta)))
            dVs = []
            delta = (1-Z[-1]**2) * np.dot( delta, self.W[1:,:].T)
            for Zi in range(len(self.nhs),0,-1):
                Vi = Zi - 1 # because X is first element of Z
                dV = np.vstack(( np.dot(np.ones((1,delta.shape[0])), delta),
                                 np.dot( Z[Zi-1].T, delta)))
                dVs.insert(0,dV)
                delta = np.dot( delta, self.Vs[Vi][1:,:].T) * (1-Z[Zi-1]**2)
            return pack(dVs,dW)

        scgresult = SCG.scg(pack(self.Vs,self.W), objectiveF, gradF,
                           xPrecision = weightPrecision,
                           fPrecision = errorPrecision,
                           nIterations = nIterations,
                           iterationVariable = self.iteration,
                           ftracep=True)

        unpack(scgresult['x'])
        self.reason = scgresult['reason']
        self.errorTrace = scgresult['ftrace']
        self.numberOfIterations = len(self.errorTrace)
        self.trained.value = True
        return self
    
    def getNumberOfIterations(self):
        return self.numberOfIterations
        
    def use(self,X,allOutputs=False):
        Zprev = self.standardizeX(X)
        Z = [Zprev]
        for i in range(len(self.nhs)):
            V = self.Vs[i]
            Zprev = np.tanh(np.dot(Zprev,V[1:,:]) + V[0:1,:])
            Z.append(Zprev)
        Y = np.dot(Zprev, self.W[1:,:]) + self.W[0:1,:]
        Y = self.unstandardizeT(Y)
        return (Y,Z[1:]) if allOutputs else Y

    def getErrorTrace(self):
        return self.errorTrace
    
    def plotErrors(self):
        plt.plot(self.errorTrace)
        plt.ylabel("Train RMSE")
        plt.xlabel("Iteration")

    def addOnes(self,X):
        return np.hstack((np.ones((X.shape[0],1)), X))

    def standardizeX(self,X):
        return (X - self.Xmeans) / self.Xstds
    def unstandardizeX(self,Xs):
        return self.Xstds * Xs + self.Xmeans
    def standardizeT(self,T):
        return (T - self.Tmeans) / self.Tstds
    def unstandardizeT(self,Ts):
        return self.Tstds * Ts + self.Tmeans

    def makeIndicatorVars(self,T):
        """ Assumes argument is N x 1, N samples each being integer class label """
        return (T == np.unique(T)).astype(int)

    def draw(self, inputNames = None, outputNames = None, gray = False):
        def isOdd(x):
            return x % 2 != 0

        # if len(self.Vs) != 1:
        #     print 'draw only defined for nets with single hidden layer'
        #     return

        W = self.Vs + [self.W]
        nLayers = len(W)

        # calculate xlim and ylim for whole network plot
        #  Assume 4 characters fit between each wire
        #  -0.5 is to leave 0.5 spacing before first wire
        xlim = max(map(len,inputNames))/4.0 if inputNames else 1
        ylim = 0
    
        for li in range(nLayers):
            ni,no = W[li].shape  #no means number outputs this layer
            if not isOdd(li):
                ylim += ni + 0.5
            else:
                xlim += ni + 0.5

        ni,no = W[nLayers-1].shape  #no means number outputs this layer
        if isOdd(nLayers):
            xlim += no + 0.5
        else:
            ylim += no + 0.5

        # Add space for output names
        if outputNames:
            if isOdd(nLayers):
                ylim += 0.25
            else:
                xlim += round(max(map(len,outputNames))/4.0)

        ax = plt.gca()

        x0 = 1
        y0 = 0 # to allow for constant input to first layer
        # First Layer
        if inputNames:
            #addx = max(map(len,inputNames))*0.1
            y = 0.55
            for n in inputNames:
                y += 1
                ax.text(x0-len(n)*0.2, y, n)
                x0 = max([1,max(map(len,inputNames))/4.0])

        for li in range(nLayers):
            Wi = W[li]
            ni,no = Wi.shape
            if not isOdd(li):
                # Odd layer index. Vertical layer. Origin is upper left.
                # Constant input
                ax.text(x0-0.2, y0+0.5, '1')
                for li in range(ni):
                    ax.plot((x0,x0+no-0.5), (y0+li+0.5, y0+li+0.5),color='gray')
                # output lines
                for li in range(no):
                    ax.plot((x0+1+li-0.5, x0+1+li-0.5), (y0, y0+ni+1),color='gray')
                # cell "bodies"
                xs = x0 + np.arange(no) + 0.5
                ys = np.array([y0+ni+0.5]*no)
                ax.scatter(xs,ys,marker='v',s=1000,c='gray')
                # weights
                if gray:
                    colors = np.array(["black","gray"])[(Wi.flat >= 0)+0]
                else:
                    colors = np.array(["red","green"])[(Wi.flat >= 0)+0]
                xs = np.arange(no)+ x0+0.5
                ys = np.arange(ni)+ y0 + 0.5
                aWi = abs(Wi)
                aWi = aWi / np.max(aWi) * 20 #50
                coords = np.meshgrid(xs,ys)
                #ax.scatter(coords[0],coords[1],marker='o',s=2*np.pi*aWi**2,c=colors)
                ax.scatter(coords[0],coords[1],marker='s',s=aWi**2,c=colors)
                y0 += ni + 1
                x0 += -1 ## shift for next layer's constant input
            else:
                # Even layer index. Horizontal layer. Origin is upper left.
                # Constant input
                ax.text(x0+0.5, y0-0.2, '1')
                # input lines
                for li in range(ni):
                    ax.plot((x0+li+0.5,  x0+li+0.5), (y0,y0+no-0.5),color='gray')
                # output lines
                for li in range(no):
                    ax.plot((x0, x0+ni+1), (y0+li+0.5, y0+li+0.5),color='gray')
                # cell "bodies"
                xs = np.array([x0 + ni + 0.5]*no)
                ys = y0 + 0.5 + np.arange(no)
                ax.scatter(xs,ys,marker='>',s=1000,c='gray')
                # weights
                Wiflat = Wi.T.flatten()
                if gray:
                    colors = np.array(["black","gray"])[(Wiflat >= 0)+0]
                else:
                    colors = np.array(["red","green"])[(Wiflat >= 0)+0]
                xs = np.arange(ni)+x0 + 0.5
                ys = np.arange(no)+y0 + 0.5
                coords = np.meshgrid(xs,ys)
                aWi = abs(Wiflat)
                aWi = aWi / np.max(aWi) * 20 # 50
                #ax.scatter(coords[0],coords[1],marker='o',s=2*np.pi*aWi**2,c=colors)
                ax.scatter(coords[0],coords[1],marker='s',s=aWi**2,c=colors)
                x0 += ni + 1
                y0 -= 1 ##shift to allow for next layer's constant input

        # Last layer output labels 
        if outputNames:
            if isOdd(nLayers):
                x = x0+1.5
                for n in outputNames:
                    x += 1
                    ax.text(x, y0+0.5, n)
            else:
                y = y0+0.6
                for n in outputNames:
                    y += 1
                    ax.text(x0+0.2, y, n)
        ax.axis([0,xlim, ylim,0])
        ax.axis('off')

class NeuralNetClassifier(NeuralNet):
    def __init__(self,ni,nhs,no):
        #super(NeuralNetClassifier,self).__init__(ni,nh,no)
        NeuralNet.__init__(self,ni,nhs,no-1)

    def train(self,X,T,
                 nIterations=100,weightPrecision=0.0001,errorPrecision=0.0001):
        if self.Xmeans is None:
            self.Xmeans = X.mean(axis=0)
            self.Xstds = X.std(axis=0)
        X = self.standardizeX(X)
        X1 = self.addOnes(X)

        self.classes = np.unique(T)
        if self.no != len(self.classes)-1:
            raise ValueError(" In NeuralNetClassifier, the number of outputs must be one less than\n the number of classes in the training data. The given number of outputs\n is %d and number of classes is %d. Try changing the number of outputs in the\n call to NeuralNetClassifier()." % (self.no, len(self.classes)))
        T = self.makeIndicatorVars(T)

        # Local functions used by gradientDescent.scg()
        def pack(Vs,W):
            # r = np.hstack([V.flat for V in Vs] + [W.flat])
            # print 'pack',len(Vs), Vs[0].shape, W.shape,r.shape
            return np.hstack([V.flat for V in Vs] + [W.flat])
        def unpack(w):
            first = 0
            numInThisLayer = self.ni
            for i in range(len(self.Vs)):
                self.Vs[i][:] = w[first:first+(numInThisLayer+1)*self.nhs[i]].reshape((numInThisLayer+1,self.nhs[i]))
                first += (numInThisLayer+1) * self.nhs[i]
                numInThisLayer = self.nhs[i]
            self.W[:] = w[first:].reshape((numInThisLayer+1,self.no))
        def objectiveF(w):
            unpack(w)
            Zprev = X
            for i in range(len(self.nhs)):
                V = self.Vs[i]
                Zprev = np.tanh(np.dot(Zprev,V[1:,:]) + V[0:1,:])
            Y = np.dot(Zprev, self.W[1:,:]) + self.W[0:1,:]
            return 0.5 * np.mean((Y - T)**2)
        def gradF(w):
            unpack(w)
            Zprev = X
            Z = [Zprev]
            for i in range(len(self.nhs)):
                V = self.Vs[i]
                Zprev = np.tanh(np.dot(Zprev,V[1:,:]) + V[0:1,:])
                Z.append(Zprev)
            Y = np.dot(Zprev, self.W[1:,:]) + self.W[0:1,:]
            delta = (Y - T) / (X1.shape[0] * T.shape[1])
            dW = np.vstack((np.dot(np.ones((1,delta.shape[0])),delta),  np.dot( Z[-1].T, delta)))
            dVs = []
            delta = (1-Z[-1]**2) * np.dot( delta, self.W[1:,:].T)
            for Zi in range(len(self.nhs),0,-1):
                Vi = Zi - 1 # because X is first element of Z
                dV = np.vstack(( np.dot(np.ones((1,delta.shape[0])), delta),
                                 np.dot( Z[Zi-1].T, delta)))
                dVs.insert(0,dV)
                delta = np.dot( delta, self.Vs[Vi][1:,:].T) * (1-Z[Zi-1]**2)
            return pack(dVs,dW)

        def objectiveF(w):
            unpack(w)
            Zprev = X
            for i in range(len(self.nhs)):
                V = self.Vs[i]
                Zprev = np.tanh(np.dot(Zprev,V[1:,:]) + V[0:1,:])
            Y = np.dot(Zprev, self.W[1:,:]) + self.W[0:1,:]
            expY = np.exp(Y)
            denom = 1 + np.sum(expY,axis=1).reshape((-1,1))
            Y = np.hstack((expY / denom, 1/denom))
            return -np.mean(T * np.log(Y))
        def gradF(w):
            unpack(w)
            Zprev = X
            Z = [Zprev]
            for i in range(len(self.nhs)):
                V = self.Vs[i]
                Zprev = np.tanh(np.dot(Zprev,V[1:,:]) + V[0:1,:])
                Z.append(Zprev)
            Y = np.dot(Zprev, self.W[1:,:]) + self.W[0:1,:]
            expY = np.exp(Y)
            denom = 1 + np.sum(expY,axis=1).reshape((-1,1))
            Y = np.hstack((expY /denom , 1.0/denom))
            delta = (Y[:,:-1] - T[:,:-1]) / (X1.shape[0] * (T.shape[1]-1))
            dW = np.vstack((np.dot(np.ones((1,delta.shape[0])),delta),  np.dot( Z[-1].T, delta)))
            dVs = []
            delta = (1-Z[-1]**2) * np.dot( delta, self.W[1:,:].T)
            for Zi in range(len(self.nhs),0,-1):
                Vi = Zi - 1 # because X is first element of Z
                dV = np.vstack(( np.dot(np.ones((1,delta.shape[0])), delta),
                                 np.dot( Z[Zi-1].T, delta)))
                dVs.insert(0,dV)
                delta = np.dot( delta, self.Vs[Vi][1:,:].T) * (1-Z[Zi-1]**2)
            return pack(dVs,dW)

        scgresult = SCG.scg(pack(self.Vs,self.W), objectiveF, gradF,
                           xPrecision = weightPrecision,
                           fPrecision = errorPrecision,
                           nIterations = nIterations,
                           iterationVariable = self.iteration,
                           ftracep=True)

        unpack(scgresult['x'])
        self.reason = scgresult['reason']
        self.errorTrace = scgresult['ftrace']
        self.numberOfIterations = len(self.errorTrace)
        self.trained.value = True
        return self
    
        Y = self.unstandardizeT(Y)
        return (Y,Z[1:]) if allOutputs else Y

    def use(self,X,allOutputs=False):
        Zprev = self.standardizeX(X)
        Z = [Zprev]
        for i in range(len(self.nhs)):
            V = self.Vs[i]
            Zprev = np.tanh(np.dot(Zprev,V[1:,:]) + V[0:1,:])
            Z.append(Zprev)
        Y = np.dot(Zprev, self.W[1:,:]) + self.W[0:1,:]
        expY = np.exp(Y)
        denom = 1 + np.sum(expY,axis=1).reshape((-1,1))
        Y = np.hstack((expY / denom, 1/denom))
        classes = self.classes[np.argmax(Y,axis=1)].reshape((-1,1))
        return (classes,Y,Z) if allOutputs else classes

    def plotErrors(self):
        plt.plot(np.exp(-self.errorTrace))
        plt.ylabel("Train Likelihood")
        plt.xlabel("Iteration")

class NeuralNetQ(NeuralNet):
    def __init__(self,ni,nh,no,inputminmax):
        NeuralNet.__init__(self,ni,nh,no)
        # print np.array(inputminmax).T
        inputminmaxnp = np.array(inputminmax)
        self.Xmeans = inputminmaxnp.mean(axis=0)
        self.Xstds = inputminmaxnp.std(axis=0)

    def train(self,X,R,Q,Y,gamma=1,
                 nIterations=100,weightPrecision=0.0001,errorPrecision=0.0001):
        X = self.standardizeX(X)
        X1 = self.addOnes(X)

        def pack(Vs,W):
            # r = np.hstack([V.flat for V in Vs] + [W.flat])
            # print 'pack',len(Vs), Vs[0].shape, W.shape,r.shape
            return np.hstack([V.flat for V in Vs] + [W.flat])
        def unpack(w):
            first = 0
            numInThisLayer = self.ni
            for i in range(len(self.Vs)):
                self.Vs[i][:] = w[first:first+(numInThisLayer+1)*self.nhs[i]].reshape((numInThisLayer+1,self.nhs[i]))
                first += (numInThisLayer+1) * self.nhs[i]
                numInThisLayer = self.nhs[i]
            self.W[:] = w[first:].reshape((numInThisLayer+1,self.no))
        def objectiveF(w):
            unpack(w)
            Zprev = X
            for i in range(len(self.nhs)):
                V = self.Vs[i]
                Zprev = np.tanh(np.dot(Zprev,V[1:,:]) + V[0:1,:])
            Y = np.dot(Zprev, self.W[1:,:]) + self.W[0:1,:]
            return 0.5 *np.mean((R+gamma*Q-Y)**2)
        def gradF(w):
            unpack(w)
            Zprev = X
            Z = [Zprev]
            for i in range(len(self.nhs)):
                V = self.Vs[i]
                Zprev = np.tanh(np.dot(Zprev,V[1:,:]) + V[0:1,:])
                Z.append(Zprev)
            Y = np.dot(Zprev, self.W[1:,:]) + self.W[0:1,:]

            nSamples = X1.shape[0]
            delta = -(R + gamma * Q - Y) / nSamples
            dW = np.vstack((np.dot(np.ones((1,delta.shape[0])),delta),  np.dot( Z[-1].T, delta)))
            dVs = []
            delta = (1-Z[-1]**2) * np.dot( delta, self.W[1:,:].T)
            for Zi in range(len(self.nhs),0,-1):
                Vi = Zi - 1 # because X is first element of Z
                dV = np.vstack(( np.dot(np.ones((1,delta.shape[0])), delta),
                                 np.dot( Z[Zi-1].T, delta)))
                dVs.insert(0,dV)
                delta = np.dot( delta, self.Vs[Vi][1:,:].T) * (1-Z[Zi-1]**2)
            return pack(dVs,dW)


        scgresult = SCG.scg(pack(self.Vs,self.W), objectiveF, gradF,
                            xPrecision = weightPrecision,
                            fPrecision = errorPrecision,
                            nIterations = nIterations,
                            iterationVariable = self.iteration,
                            ftracep=True)

        unpack(scgresult['x'])
        self.reason = scgresult['reason']
        self.errorTrace = scgresult['ftrace']
        self.numberOfIterations = len(self.errorTrace)
        self.trained.value = True
        return self

    def use(self,X,allOutputs=False):
        X1 = self.addOnes(self.standardizeX(X))
        Z = np.tanh(np.dot( X1, self.V ))
        Z1 = self.addOnes(Z)
        Y = np.dot( Z1, self.W )
        return (Y,Z) if allOutputs else Y

######################################################################
# Utilities. Should be in own module (file).

# Build matrix of labeled time windows, each including nlags consecutive
# samples, with channels concatenated.  Keep only segments whose nLags
# targets are the same. Assumes target is in last column of data.
def segmentize(data,nLags):
    nSamples,nChannels = data.shape
    nSegments = nSamples-nLags+1
    # print 'segmentize, nSamples',nSamples,'nChannels',nChannels,'nSegments',nSegments
    # print 'data.shape',data.shape,'nLags',nLags
    segments = np.zeros((nSegments, (nChannels-1) * nLags + 1))
    k = 0
    for i in range(nSegments):
        targets = data[i:i+nLags,-1]
        # In previous versions had done this...
        #    to throw away beginning and end of each trial, and between trials
        #       if targets[0] != 0 and targets[0] != 32: 
        #            # keep this row
        if (np.all(targets[0] == targets)):
            allButTarget = data[i:i+nLags,:-1]
            # .T to keep all from one channel together
            segments[k,:-1] = allButTarget.flat
            segments[k,-1] = targets[0]
            k += 1
    return segments[:k,:]

def confusionMatrix(actual,predicted,classes):
    nc = len(classes)
    confmat = np.zeros((nc,nc))
    for ri in range(nc):
        trues = actual==classes[ri]
        for ci in range(nc):
            confmat[ri,ci] = np.sum(predicted[trues] == classes[ci]) / float(np.sum(trues))
    return confmat*100

def printConfusionMatrix(percents,classes):
    print '   ',
    for i in classes:
        print '%4d:' % (i),
    print
    for i,t in enumerate(classes):
        print '%2d: ' % (t),
        for i1,t1 in enumerate(classes):
            if percents[i,i1] == 0:
                print '  0  ',
            else:
                print '%5.1f' % (percents[i,i1]),
        print 


def partition(data,trainPercent,classification=True,nTargets=1):
    """Usage: Xtrain,Train,Xtest,Ttest = partition(X,80,classification=True)
    or Xtrain,Train,Xtest,Ttest = partition(X,80,classification=False,nTargets=3)
      X is nSamples x nFeatures.
      If classification=True, last column of X is target class as integer.
      If classification=False, last nTargets columns are targets for regression."""
    if classification == 1:
        # classifying, so partition data according to target class
        targets = data[:,-1]
        classes = np.unique(targets)
        print classes
        trainrows = []
        testrows = []
        for c in classes:
            cRows = np.where(targets == c)[0]
            # np.random.shuffle(cRows) # randomly reorder, in place
            firstTestRow = int(0.5+len(cRows) * trainPercent/100.0)
            trainrows += list(cRows[:firstTestRow])
            testrows += list(cRows[firstTestRow:])
        trainrows.sort()
        testrows.sort()
        trainrows = np.array(trainrows)
        testrows = np.array(testrows)
        Xtrain = data[trainrows,:-1]
        Ttrain = data[trainrows,-1:] #preserves 2d
        Xtest = data[testrows,:-1]
        Ttest = data[testrows,-1:]

        # plt.clf();plt.plot(data[:,-1])
        # plt.plot(trainrows,Ttrain,'ro');
        # plt.plot(testrows,Ttest,'go')

    else:
        # regression, so do not partition according to targets.
        n = data.shape[0]
        firstTestRow = int(0.5 + n*trainPercent/100.0)
        #rows = np.arange(n)
        Xtrain = data[:firstTestRow,:-nTargets]
        Ttrain = data[:firstTestRow,-nTargets:]
        Xtest =  data[firstTestRow:,:-nTargets]
        Ttest = data[firstTestRow:,-nTargets:]
    return Xtrain,Ttrain,Xtest,Ttest

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

    def experiment(hs,nReps,nIter,X,T,Xtest,Ytest):
        results = []
        for i in range(nReps):
            nnet = NeuralNet(1,hs,1)
            nnet.train(X,T,weightPrecision=1.e-20,errorPrecision=1.e-20,nIterations=nIter)
            # print "SCG stopped after",nnet.getNumberOfIterations(),"iterations:",nnet.reason
            (Y,Z) = nnet.use(X, allOutputs=True)
            Ytest = nnet.use(Xtest)
            rmsetrain = np.sqrt(np.mean((Y-T)**2))
            rmsetest = np.sqrt(np.mean((Ytest-Ttest)**2))
            results.append([rmsetrain,rmsetest])
        return results

    results = []
    # hiddens [ [5]*i for i in range(1,6) ]
    hiddens = [[12], [6,6], [4,4,4], [3,3,3,3], [2,2,2,2,2,2]]
    for hs in hiddens:
        r = experiment(hs,10,5000,X,T,Xtest,Ttest)
        r = np.array(r)
        means = np.mean(r,axis=0)
        stds = np.std(r,axis=0)
        results.append([hs,means,stds])
        print hs, means,stds
        rmseMeans = np.array([x[1].tolist() for x in results])
        plt.clf()
        plt.plot(rmseMeans,'o-')
        plt.draw()
        
    # # nnet = NeuralNet(1,(5,5,5,5,5,5,5,,5,),1)
    # # nnet = NeuralNet(1,(5,),1)
    # # nnet = NeuralNet(1,(5,4,3,2),1)
    # # nnet = NeuralNet(1,(5,5,5,5,5,2),1)
    # # nnet = NeuralNet(1,(10,2,10),1)
    # # nnet = NeuralNet(1,(10,1,10),1)
    # # nnet = NeuralNet(1,(5,5,),1)
    # nnet = NeuralNet(1,(5,2,),1)
    
    # nnet.train(X,T,weightPrecision=1.e-20,errorPrecision=1.e-20,nIterations=10000)
    # print "SCG stopped after",nnet.getNumberOfIterations(),"iterations:",nnet.reason
    # (Y,Z) = nnet.use(X, allOutputs=True)
    # Ytest = nnet.use(Xtest)
    # print "Final RMSE: train", np.sqrt(np.mean((Y-T)**2)),"test",np.sqrt(np.mean((Ytest-Ttest)**2))
    
    # # print 'Inputs, Targets, Estimated Targets'
    # # print np.hstack((X,T,Y))

    # nPlotRows = 3 + len(nnet.nhs)
    # plt.figure(2)
    # plt.clf()
    # plt.subplot(nPlotRows,1,1)
    # nnet.plotErrors()
    # plt.title('Regression Example')
    # plt.subplot(nPlotRows,1,2)
    # plt.plot(X,T)
    # plt.plot(X,Y)
    # plt.legend(('Train Target','Train NN Output'),loc='lower right',
    #            prop={'size':9})
    # plt.subplot(nPlotRows,1,3)
    # plt.plot(Xtest,Ttest)
    # plt.plot(Xtest,Ytest)
    # plt.legend(('Test Target','Test NN Output'),loc='lower right',
    #            prop={'size':9})
    # colors = ('blue','green','red','black','cyan','orange')
    # for i in range(len(nnet.nhs)):
    #     plt.subplot(nPlotRows,1,i+4)
    #     plt.plot(X,Z[i]) #,color=colors[i])
    #     plt.ylabel('Hidden Unit Outputs')
    #     plt.text(8,0, 'Layer {}'.format(i+1))

    # plt.figure(3)
    # plt.clf()
    # nnet.draw(['x'],['sine'])


    print '\n------------------------------------------------------------'
    print "Classification Example: XOR, approximate f(x1,x2) = x1 xor x2"
    print '                        Using neural net with 2 inputs, 3 hidden units, 2 outputs'
    X = np.array([[0,0],[1,0],[0,1],[1,1]])
    T = np.array([[1],[2],[2],[1]])
    nnet = NeuralNetClassifier(2,(3,3,3),2)
    nnet.train(X,T,weightPrecision=1.e-8,errorPrecision=1.e-8,nIterations=100)
    print "SCG stopped after",nnet.getNumberOfIterations(),"iterations:",nnet.reason
    (classes,y,Z) = nnet.use(X, allOutputs=True)
    

    print 'X(x1,x2), Target Classses, Predicted Classes'
    print np.hstack((X,T,classes))

    print "Hidden Outputs"
    print Z
    
    # plt.ion()
    plt.figure(1)
    plt.clf()
    plt.subplot(2,1,1)
    nnet.plotErrors()
    plt.title('Classification Example')
    plt.subplot(2,1,2)
    nnet.draw(['x1','x2'],['xor'])
