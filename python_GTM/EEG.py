import numpy as np
import scipy as sp
import matplotlib as pyplot
import random
import sys

class GTM:
    def __init__(self,dim_latent, nlatent, dim_data, ncentres, rbfunc, prior):
        # Input to functions is data
        self.type = "gtm"
        self.nin = dim_data
        self.dim_latent = dim_latent
	self.rbfnet = RBF(dim_latent, ncentres, dim_data, rbfunc, 'linear', prior, 0.1)      # Create RBF network
        #print 'w2$$$$$$', self.rbfnet.nin
        self.rbfnet.mask = RBFPRIOR(rbfunc, dim_latent, ncentres, dim_data,1,1)             # Mask all but output weights
        self.gmmnet = GMM(dim_data, nlatent, 'spherical')                                    # Create field for GMM output model
        self.X = []

class RBF:
    def __init__(self,nin, nhidden, nout, rbfunc, outfunc, prior, beta):
        self.type = "rbf"
        self.nin = nin
        self.nhidden = nhidden
        self.nout = nout
        self.actfn = rbfunc
        self.outfn = outfunc
        self.nwts = nin*nhidden + (nhidden + 1)*nout
        print 'nwts',self.nwts
        if rbfunc == 'gaussian':
            self.nwts = self.nwts + nhidden    # Extra weights for width parameters
        self.alpha = prior
        self.beta = beta
        print 'nwts',self.nwts
        np.random.seed(123)
        w = np.random.randn(1, self.nwts)
        print 'W shape',w.shape
        self=RBFUNPAK(self, w)
        #print 'w2**** ',self.w2
        # Make widths equal to one
        if rbfunc =='gaussian':
            self.wi = np.ones((1, nhidden))
	'''
        if self.outfn == 'neuroscale':
            self.mask = RBFPRIOR(rbfunc, nin, nhidden, nout)'''

class RBFUNPAK:
    def __init__(self, net, w):
        self.nin = net.nin
        self.nhidden = net.nhidden
        self.nout = net.nout
        self.actfn = net.actfn
        mark1 = self.nin*self.nhidden
        a = w[:,0:mark1]
        net.c = np.reshape(a, (self.nhidden, self.nin))
        if self.actfn == 'gaussian':
            mark2 = mark1 + self.nhidden
            a = w[:,mark1:mark2]
            net.wi = np.reshape(a, (1, self.nhidden))
        else :
            mark2 = mark1
            net.wi = []
        mark3 = mark2 + self.nhidden*self.nout
        a = w[:, mark2:mark3]
        net.w2 = np.reshape(a, (self.nhidden, self.nout))
        mark4 = mark3 + self.nout
        a = w[:, mark3:mark4]
        net.b2 = np.reshape(a, (1, self.nout))


class RBFPRIOR:
    def __init__(self, rbfunc, nin, nhidden, nout, aw2, ab2):
        self.rbfunc = rbfunc
        self.nin = nin
        self.nhidden = nhidden
        self.nout = nout
        self.aw2 = aw2
        self.ab2 = ab2
        nwts_layer2 = nout + (nhidden *nout)
        if rbfunc ==  'gaussian':
            nwts_layer1 = nin*nhidden + nhidden 
        if rbfunc == 'tps' or rbfunc == 'r4logr':
            nwts_layer1 = nin*nhidden;
        else :
            print 'Undefined activation function'
        nwts = nwts_layer1 + nwts_layer2

        # Make a mask only for output layer
        mask = np.vstack((np.zeros((nwts_layer1, 1)), np.ones((nwts_layer2, 1))))
        #Construct prior
        indx = np.zeros((nwts, 2))
        print indx.shape
        mark2 = nwts_layer1 + (nhidden * nout)
        print mark2," ",nwts_layer2," ",nwts
        for i in range(nwts_layer1,mark2):
            indx[i, 0] = 1
        for i in range(mark2,nwts):
            indx[i, 1] = 1
        class Prior:
            def __init__(self,indx,alpha):
                self.index = indx
                self.alpha = alpha
        a = (np.hstack((aw2, ab2)).T)
        prior = Prior(indx,a)

class GMM:
    def __init__(self,dim, ncentres, covar_type):
        self.type = 'gmm'
        self.nin = dim;
        self.ncentres = ncentres
        self.covar_type = covar_type
        self.priors = np.ones((1,self.ncentres)) / self.ncentres
        self.centres = np.random.randn(self.ncentres, self.nin)
        #covariance type = spherical
        self.covars = np.ones((1, self.ncentres))
        self.nwts = self.ncentres + self.ncentres*self.nin + self.ncentres

class GTMINIT:
    def __init__(self,net, rbf_width_factor, data, samp_type, latent_dim_points, rbf_dim_points):
        self.nlatent = net.gmmnet.ncentres
        self.nhidden = net.rbfnet.nhidden 
        # Sample type is regular and latent_dim=2
        #print 'w2$$$ ',net.rbfnet.c
        net.X = self.gtm_rctg(latent_dim_points)
        net.rbfnet.c = self.gtm_rctg(rbf_dim_points)
        net.rbfnet.wi = RBFSETFW(net.rbfnet, rbf_width_factor).wi
        Pcaobj = PCA(data)
        PCcoeff = Pcaobj.PCcoeff
        PCvec = Pcaobj.PCvec
        print '*&^*@#',PCcoeff.shape, PCvec.shape
        # Scale PCs by eigenvalues
        A = np.dot(PCvec[:, 0:net.dim_latent],np.diag(np.sqrt(PCcoeff[0:net.dim_latent])))
        print 'XXXXXX',net.rbfnet.c
        obj = RBFFWD(net.rbfnet, net.X)
        temp = obj.a
        Phi = obj.z
        normX = np.dot((net.X - np.dot(np.ones(net.X.shape),np.diag(np.mean(net.X,0)))),np.diag(np.divide(1,np.std(net.X,0))))
        print Phi.shape
        print "setg",normX.shape, A.T.shape
        net.rbfnet.w2 = np.linalg.lstsq(Phi , np.dot(normX,A.T))

    def gtm_rctg(self, samp_size):
        xDim = samp_size
        yDim = samp_size
        # Produce a grid with the right number of rows and columns
        X, Y = np.meshgrid(np.linspace(0,(xDim-1),xDim), np.linspace(0,(yDim-1),yDim))
        print 'X',X.shape
        print 'Y',Y.shape
        # Change grid representation 
        sample = np.zeros((xDim**2,2))
        sample[:,0:1]=np.reshape(X,(xDim**2,1),1)
        sample[:,1:2]=np.reshape(Y,(yDim**2,1),1)
        # Shift grid to correct position and scale it
        maxXY= np.max(sample,0);
        sample[:,0:1] = 2*(sample[:,0:1] - maxXY[0]/2)/maxXY[0]
        sample[:,1:2] = -2*(sample[:,1:2] - maxXY[1]/2)/maxXY[1]
        return sample

class RBFSETFW:
# Sets the widths of the basis functions of the RBF network NET
    def __init__(self, net, scale):
        print '%%%%%%%', net.c
        self.cdist = DIST2(net.c, net.c).n2
        if scale > 0.0:
        # Set variance of basis to be scale times average distance to nearest neighbour
            self.cdist = self.cdist + sys.float_info.max*np.eye(net.nhidden)
            self.widths = scale*np.mean((self.cdist).min(axis=0))
        else:
            self.widths = np.max(self.cdist)
        self.wi = self.widths * np.ones(net.wi.shape)

class DIST2:
#Calculate squared euclidean distance between x and c
    def __init__(self, x, c):
        self.ndata = x.shape[0]
        self.dimx = x.shape[1]
        self.ncentres = c.shape[0]
        self.dimc = c.shape[1]
        self.n2 = (np.ones((self.ncentres, 1)) * sum((x**2).T, 1)).T + np.ones((self.ndata, 1)) * sum((c**2).T,1) -  2*(np.dot(x,c.T))
        # Rounding errors occasionally cause negative entries in n2
        self.n2 = np.where(self.n2>=0,self.n2,0)


class PCA:
    def __init__(self, data):
        print 'awriha',data.shape[1],np.cov(data).shape
        self.N=data.shape[1]
        a = eigdec(np.cov(data.T), self.N)
        self.PCcoeff = a.evals
        self.PCvec = a.evec

class eigdec:
    def __init__(self, x, N):
        if N/x.shape[1]>0.04:
            self.evals, self.evec = np.linalg.eig(x)
        else:
            self.evals, self.evec = sp.sparse.linalg.eigs(x, 'LM')
        print N,'55555555555555  ',self.evals.shape
        self.evals = -1*self.evals[0:N]
        self.evec = self.evec[:, 0:N]

class RBFFWD:
# Takes a network data structure NET and a matrix X of input vectors and forward propagates the inputs through the network to generate a matrix A of output vectors
    def __init__(self, net, x):
        self.ndata = x.shape[0]
        self.data_dim = x.shape[1]
        self.n2 = DIST2(x, net.c).n2
        print self.ndata
        print net.wi
        self.wi2 = np.dot(np.ones((self.ndata, 1)), (2 * net.wi))
        self.z = np.exp(-np.divide(self.n2,self.wi2))
        print self.z.shape, " ,",net.w2.shape,",",self.ndata,",",net.b2.shape
        self.a = np.dot(self.z,net.w2) + np.dot(np.ones((self.ndata, 1)),net.b2)

###################################################################################################################
data=np.loadtxt('classData.txt')

# Generate the grid of latent points

latent_dim_points = 15  # Number of latent points in each dimension
nlatent = latent_dim_points**2  # Number of latent points
data_dim = 4
latent_dim = 2

rbf_dim_points = 4
num_rbf_centres = rbf_dim_points**2
rbf_width_factor = 1

# Create and Intialize GTM model

net = GTM(latent_dim, nlatent, data_dim, num_rbf_centres,'gaussian', 0.1)
net = GTMINIT(net, rbf_width_factor, data, 'regular', latent_dim_points, rbf_dim_points)

