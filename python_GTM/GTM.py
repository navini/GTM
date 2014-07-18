
import numpy as np
from scipy import linalg
from scipy import sparse
from matplotlib import pyplot as plt
import random
import sys
import math


def gtm_rctg(samp_size):
    xDim = samp_size
    yDim = samp_size
    # Produce a grid with the right number of rows and columns
    X, Y = np.meshgrid(np.linspace(0,(xDim-1),xDim), np.linspace(0,(yDim-1),yDim))
    # Change grid representation 
    sample = np.zeros((xDim**2,2))
    sample[:,0:1]=np.reshape(X,(xDim**2,1),1)
    sample[:,1:2]=np.reshape(Y,(yDim**2,1),1)
    # Shift grid to correct position and scale it
    maxXY= np.max(sample,0);
    sample[:,0:1] = 2*(sample[:,0:1] - maxXY[0]/2)/maxXY[0]
    sample[:,1:2] = -2*(sample[:,1:2] - maxXY[1]/2)/maxXY[1]
    return sample

def dist(x,c):
#Calculate squared euclidean distance between x and c
    ndata = x.shape[0]
    dimx = x.shape[1]
    ncentres = c.shape[0]
    dimc = c.shape[1]
    x2 = np.reshape(sum(np.square(x).T),(1,ndata))
    c2 = np.reshape(sum(np.square(c).T),(1,ncentres))
    n2 = (np.dot(np.ones((ncentres, 1)) , x2)).T +np.dot(np.ones((ndata, 1)),c2) -  2*(np.dot(x,c.T))
    # Rounding errors occasionally cause negative entries in n2
    n2 = np.where(n2>=0,n2,0)
    return n2

def rbfsetfw(rbf_c,rbf_w,scale):   
# Set the function widths of the RBF network
    cdist = dist(rbf_c,rbf_c)
    if scale > 0.0:
    # Set variance of basis to be scale times average distance to nearest neighbour
        cdist = cdist + sys.float_info.max*np.eye(rbf_c.shape[0])
        widths = scale*np.mean((cdist).min(axis=0))
    else:
        widths = np.max(cdist)
    wi = widths * np.ones(rbf_w.shape)
    return wi

def eigdec(x, N):
    temp_evals, temp_evec = np.linalg.eig(x)
    #temp_evals = np.sort(temp_evals)
    print 'temp_evals'
    print temp_evals
    evals = np.sort(-temp_evals)
    perm = np.argsort(-temp_evals)
    print 'evals', evals
    print 'perm',perm
    evals = -evals[0:N]
    evec = np.zeros((temp_evec.shape[0],N), dtype=complex)
    if (evals[0:N] == temp_evals[0:N]).all():
        evec = temp_evec[:,0:N]
    else:
        for i in range(N):
            evec[:,i] = temp_evec[:,perm[i]] 
    return evals, evec   

def rbffwd(c, x, wi, w2, b2):
    nx = x.shape[0]
    ndim = x.shape[1]
    n2 = dist(x, c)
    # Calculate width factors: net.wi contains squared widths
    wi2 = np.ones((nx, 1)) * (2 * wi)
    # Now compute the activations
    z = np.exp(-(n2/wi2))
    a = np.dot(z,w2) + np.dot(np.ones((nx, 1)),b2)
    return a, z, n2

def gmmactive(nin, covars, gmm_c, ncentres, data):
    a = np.zeros((data.shape[0], ncentres))
    # Calculate squared norm matrix, of dimension (ndata, ncentres)
    n2 = dist(data, gmm_c)/10000
    # Calculate width factors
    wi2 = np.dot(np.ones((data.shape[0], 1)), (2 * covars))
    #normal = (math.pi* wi2)**(nin/2)
    # Now compute the activations
    #a = np.exp(-(n2/wi2))/ normal
    a = np.exp(-(n2/wi2))
    return a
    

def gtmpost(nin, covars, gmm_c, ncentres, priors, rbf_c, X, wi, w2, b2, data):
    gmm_centres = rbffwd(rbf_c, X, wi, w2, b2)
    a = gmmactive(nin, covars, gmm_c, ncentres, data)
    post = (np.dot(np.ones((data.shape[0], 1)),priors))*a
    s = np.reshape(post.sum(axis=1).T,(data.shape[0],1))
    # Set any zeros to one before dividing
    s = np.where(s>0,s,1)
    post = post/np.dot(s,np.ones((1, ncentres)))
    return post, a

#def drawSignal(classData, plotData, xmin, ymin, xmax, ymax, color):


def _get_limits( ax ):
    """ Return X and Y limits for the passed axis as [[xlow,xhigh],[ylow,yhigh]]
    """
    return [list(ax.get_xlim()), list(ax.get_ylim())]

def _set_limits( ax, lims ):
    """ Set X and Y limits for the passed axis
    """
    ax.set_xlim(*(lims[0]))
    ax.set_ylim(*(lims[1]))
    return

def pre_zoom( fig ):
    """ Initialize history used by the re_zoom() event handler.
        Call this after plots are configured and before plt.show().
    """
    global oxy
    oxy = [_get_limits(ax) for ax in fig.axes]
    # :TODO: Intercept the toolbar Home, Back and Forward buttons.
    return

def re_zoom(event):
    """ plt event handler to zoom all plots together, but permit them to
        scroll independently.  Created to support eyeball correlation.
        Use with 'motion_notify_event' and 'button_release_event'.
    """
    global oxy
    for ax in event.canvas.figure.axes:
        navmode = ax.get_navigate_mode()
        if navmode is not None:
            break
    scrolling = (event.button == 1) and (navmode == "PAN")
    if scrolling:                   # Update history (independent of event type)
        oxy = [_get_limits(ax) for ax in event.canvas.figure.axes]
        return
    if event.name != 'button_release_event':    # Nothing to do!
        return
    # We have a non-scroll 'button_release_event': Were we zooming?
    zooming = (navmode == "ZOOM") or ((event.button == 3) and (navmode == "PAN"))
    if not zooming:                 # Nothing to do!
        oxy = [_get_limits(ax) for ax in event.canvas.figure.axes]  # To be safe
        return
    # We were zooming, but did anything change?  Check for zoom activity.
    changed = None
    print '66666333311115555 ',ax.get_xlim()
    zoom = [[0.0,0.0],[0.0,0.0]]    # Zoom from each end of axis (2 values per axis)
    '''for i, ax in enumerate(event.canvas.figure.axes): # Get the axes
        # Find the plot that changed
        nxy = _get_limits(ax)
        if (oxy[i] != nxy):         # This plot has changed
            changed = i
            # Calculate zoom factors
            for j in [0,1]:         # Iterate over x and y for each axis
                # Indexing: nxy[x/y axis][lo/hi limit]
                #           oxy[plot #][x/y axis][lo/hi limit]
                width = oxy[i][j][1] - oxy[i][j][0]
                # Determine new axis scale factors in a way that correctly
                # handles simultaneous zoom + scroll: Zoom from each end.
                zoom[j] = [(nxy[j][0] - oxy[i][j][0]) / width,  # lo-end zoom
                           (oxy[i][j][1] - nxy[j][1]) / width]  # hi-end zoom
            break '''                  # No need to look at other axes
    '''if changed is not None:
        for i, ax in enumerate(event.canvas.figure.axes): # change the scale
            if i == changed:
                continue
            for j in [0,1]:
                width = oxy[i][j][1] - oxy[i][j][0]
                nxy[j] = [oxy[i][j][0] + (width*zoom[j][0]),
                          oxy[i][j][1] - (width*zoom[j][1])]
            _set_limits(ax, nxy)
        event.canvas.draw()         # re-draw the canvas (if required)
        pre_zoom(event.canvas.figure)   # Update history'''
    return
# End re_zoom()
'''
def zoom_factory(ax,base_scale = 2.):
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        # set the range
        cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
            print event.button
        # set new limits
        ax.set_xlim([xdata - cur_xrange*scale_factor,
                     xdata + cur_xrange*scale_factor])
        ax.set_ylim([ydata - cur_yrange*scale_factor,
                     ydata + cur_yrange*scale_factor])
        ax.figure.canvas.draw() # force re-draw
 
    fig = ax.get_figure() # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect('scroll_event',zoom_fun)
 
    #return the function
    return zoom_fun'''

#*******************************************************************************************************************************************88
    	
data=np.loadtxt('P4all3protS11smoothed.txt')
print '###',data.shape
data_features = data[:,2:212]
data_classes = data[:,1:2]

#data_dim = data_features.shape[1]  # Data dimension
data_dim = 210
print data_dim
data_samples = data_features.shape[0]

# Create the latent space

latent_dim_points = 15  # Number of latent points in each dimension
nlatent = latent_dim_points**2  # Number of latent points
latent_dim = 2  # Dimension of the latent space


#plt.plot(data_features[0,:],)
#plt.show()

########## Create the GTM 

gtm_nin = data_dim     # Number of input units to the GTM
gtm_dimLatent = latent_dim   #Latent dimension of the GTM
gtm_X = []      # Sample latent points in the GTM

########### Create the RBF space

rbf_dim_points = 4
rbf_width_factor = 1
num_rbf_centres = rbf_dim_points**2
rbf_nhidden = num_rbf_centres  # Number of hidden units
rbf_nin = latent_dim     # Number of input units
rbf_nout = data_dim    # Number of output units
rbf_nwts = (rbf_nin+1)*rbf_nhidden + (rbf_nhidden+1)*rbf_nout   # Number of total weights and biases
rbf_actfn = 'gaussian'  # For radially symmetric gaussian functions as the activation function, the implementation is based on gaussian
rbf_outfn = 'linear'    # Defines the output error function, linear for linear outputs and SoS error
# Separate the weights into its components, initialized to random normal values
w = np.random.randn(1,rbf_nwts)
mark1 = rbf_nin*rbf_nhidden
mark2 = mark1 + rbf_nhidden
mark3 = mark2 + rbf_nhidden*rbf_nout
mark4 = mark3 + rbf_nout
rbf_centres = np.reshape(w[:,0:mark1],(rbf_nhidden,rbf_nin))   # RBF centres   
rbf_wi = np.reshape(w[:,mark1:mark2],(1,rbf_nhidden))   # Squared widths needed for 'gaussian' type activation function
rbf_w2 = np.reshape(w[:,mark2:mark3],(rbf_nhidden,rbf_nout))  # 2nd layer weight matrix 
rbf_b2 = np.reshape(w[:,mark3:mark4],(1,rbf_nout))   # 2nd layer bias vector
#Make widths equal to one
rbf_wi = np.ones((1,rbf_nhidden))
'''
print 'rbf_centres',rbf_centres
print 'rbf_wi',rbf_wi
print 'rbf_w2',rbf_w2
print 'rbf_b2',rbf_b2'''

########## Create the Gaussian Mixture model

gmm_nin = data_dim
gmm_ncentres = nlatent
gmm_covartype = 'spherical'  # Allows single variance parameter for each component(stored as a vector)
gmm_priors = np.ones((1,gmm_ncentres))/gmm_ncentres    #Priors are the mixing coefficients. Initialized to equal values summing to one, and the covariance are all the identity matrices
gmm_centres = np.random.randn(gmm_ncentres,gmm_nin)   # Initialized from tandom normal distribution
gmm_covars = np.ones((1,gmm_ncentres))
gmm_nwts = gmm_ncentres + gmm_ncentres*gmm_nin + gmm_ncentres    # Total number of weights and biases
'''
print('gmm priors',gmm_priors)
print(gmm_priors.shape)
print('gmm centres',gmm_centres)
print('gmm nwts',gmm_nwts)'''

######## Initialize the weights and latent samples in the GTM. Generates a sample of latent data points and sets the centres and widths of the RBF network

gtm_X = gtm_rctg(latent_dim_points)
rbf_centres = gtm_rctg(rbf_dim_points)
rbf_wi = rbfsetfw(rbf_centres, rbf_wi, rbf_width_factor)
# Calculate the principal components
print 'covariance'
print np.cov(data_features.T)[0:5,0:5]
PCcoeff, PCvec = eigdec(np.cov(data_features.T), data_features.shape[1])
PCcoeff = np.reshape(PCcoeff,(1,PCcoeff.shape[0]))
print 'PCcoeff'
print PCcoeff
print 'PCvec'
print PCvec[0:10,0:10]
print 'sqrt'
print np.diag(np.asarray(np.sqrt(PCcoeff[:,0:gtm_dimLatent])).reshape(-1)).shape
print PCvec[:, 0:gtm_dimLatent].shape
A = np.dot(PCvec[:, 0:gtm_dimLatent],np.diag(np.asarray(np.sqrt(PCcoeff[:,0:gtm_dimLatent])).reshape(-1)))
print 'A'
print A
# Forward prpagate through the RBF network
a , Phi, dist1 = rbffwd(rbf_centres, gtm_X, rbf_wi, rbf_w2, rbf_b2)
# Normalise X to ensure 1:1 mapping of variances and calculate weights as solution of Phi*W = normX*A'
normX = np.dot((gtm_X - np.dot(np.ones(gtm_X.shape),np.diag(np.mean(gtm_X,0)))),np.diag(1/np.std(gtm_X,0)))
g = np.dot(normX,A.T)
x_sol, residues, rank, singV = linalg.lstsq(Phi,g)            
rbf_w2 = np.asarray(x_sol)
# Bias is mean of target data
rbf_b2 = np.reshape(np.mean(data_features,axis=0),(data_dim,1)).T
# Must also set initial value of variance. Find average distance between nearest centres. Ensure that distance of centre to itself is excluded by setting diagonal entries to realmax
print 'rbf_b2',rbf_b2.shape
gmm_centres, not_used1, not_used2 = rbffwd(rbf_centres, gtm_X, rbf_wi, rbf_w2, rbf_b2)
#print  'gmm_centres',gmm_centres
d = dist(gmm_centres, gmm_centres) + np.matrix(np.identity(gmm_ncentres))*sys.float_info.max
'''print dist(gmm_centres, gmm_centres)
print np.mean(dist(gmm_centres, gmm_centres).min(axis=0))/2
print np.mean(d.min(axis=0))/2
print '%%%%%',np.matrix(np.identity(gmm_ncentres))*sys.float_info.max
print 'd',d'''
sigma = np.mean(d.min(axis=0))/2
print 'Sigma1',sigma
# Now set covariance to minimum of this and next largest eigenvalue
if gtm_dimLatent < data.shape[1]:
  sigma = min(sigma, PCcoeff[:,gtm_dimLatent:gtm_dimLatent+1]);

print 'Sigma*****', sigma
gmm_covars = sigma*np.ones((1, gmm_ncentres))



############ Training the model with EM algorithm #########################################

niters = 30
ND = data_dim * data_samples 
gmm_centres, Phi, l = rbffwd(rbf_centres, gtm_X, rbf_wi, rbf_w2, rbf_b2)
Phi = np.hstack((Phi, np.ones((gtm_X.shape[0], 1))))
PhiT = Phi.T
K, Mplus1 = Phi.shape


print 'gmm_covars',gmm_covars,'\n'
print 'gmm_centres',gmm_centres,'\n'
print 'gmm_priors',gmm_priors,'\n'
print 'rbf_centres',rbf_centres,'\n'
print 'gtm_X',gtm_X,'\n'
print 'rbf_wi',rbf_wi,'\n'
print 'rbf_w2',rbf_w2,'\n'
print 'rbf_b2',rbf_b2,'\n'
print 'data_features',data_features,'\n'


for n in range(niters):

    # Calculate responsibilities
    R, act = gtmpost(gmm_nin, gmm_covars, gmm_centres, gmm_ncentres, gmm_priors, rbf_centres, gtm_X, rbf_wi, rbf_w2, rbf_b2, data_features)
    # Calculate matrix be inverted (Phi'*G*Phi + alpha*I in the papers).
    # Sparse representation of G normally executes faster and saves memory
    A = np.dot(PhiT,np.dot(sparse.spdiags(np.sum(R,axis=0).T, 0, K, K).todense(),Phi))
    W = np.linalg.solve(A,np.dot(PhiT,np.dot(R.T,data_features)))
    rbf_w2 = W[0:rbf_nhidden, :]
    print rbf_w2.shape
    rbf_b2 = W[rbf_nhidden:rbf_nhidden+1, :]
    print rbf_b2.shape
    d = dist(data_features, np.dot(Phi,W))
   
    # Calculate new value for beta
    gmm_covars = np.dot(np.ones((1, gmm_ncentres)),(sum(sum(d*R))/ND))
    print gmm_covars.shape



forMM, notused = gtmpost(gmm_nin, gmm_covars, gmm_centres, gmm_ncentres, gmm_priors, rbf_centres, gtm_X, rbf_wi, rbf_w2, rbf_b2, data_features)


means = np.dot(forMM, gtm_X)
print gtm_X.shape
print '$$',means.shape
# Mode is maximum responsibility
max_index = np.argmax(forMM, axis=1)
modes = gtm_X[max_index, :]
classesBN = np.where(data_classes==0)[0]
classesPN = np.where(data_classes== -1)[0]
classesDN = np.where(data_classes== -2)[0]

fig = plt.figure(figsize=(9,9))

pre_zoom(fig)
axes1 = fig.add_axes([0.1, 0.1, 0.8,0.8]) # main axes

xmin, xmax = axes1.get_xlim()
print '%$%$%$%$%$%$%$%$%$%$%$%', xmin,'    ',xmax


for i in classesBN:
    #print means[i,0]," , ", means[i,1]
    #axes2 = fig.add_axes([(means[i,0]-(-0.006))/0.016*0.8+0.1-0.01, (means[i,1]-(-0.008))/0.016*0.8+0.1-0.01, 0.05,0.05])
    #axes2.set_alpha(0.0)
    axes1.plot(means[i,0], means[i,1], 'k.')
    #bn = axes2.plot(data_features[i,:],'r')
    #axes2.legend()
    #axes2.axis('off')

for i in classesPN:
    #print means[i,0]," , ", means[i,1]
    #axes2 = fig.add_axes([(means[i,0]-(-0.006))/0.016*0.8+0.1-0.01, (means[i,1]-(-0.008))/0.016*0.8+0.1-0.01, 0.05,0.05])
    #axes2.set_alpha(0.0)
    axes1.plot(means[i,0], means[i,1], 'k.')
   # pn = axes2.plot(data_features[i,:],'y')
   # axes2.axis('off')

for i in classesDN:
    #print means[i,0]," , ", means[i,1]
    #axes2 = fig.add_axes([(means[i,0]-(-0.006))/0.016*0.8+0.1-0.01, (means[i,1]-(-0.008))/0.016*0.8+0.1-0.01, 0.05,0.05])
    #axes2.set_alpha(0.0)
    axes1.plot(means[i,0], means[i,1], 'k.')
    #dn = axes2.plot(data_features[i,:],'c')
    #axes2.axis('off')
    #plt.plot(means[i,0], means[i,1], 'ro')
classesB = np.where(data_classes==1)[0]
classesP = np.where(data_classes==2)[0]
classesD = np.where(data_classes==3)[0]

for i in classesB:
    #axes2 = fig.add_axes([(means[i,0]-(-0.006))/0.016*0.8+0.1-0.01, (means[i,1]-(-0.008))/0.016*0.8+0.1-0.01, 0.05,0.05])
    #axes2.set_alpha(0.0)
    axes1.plot(means[i,0], means[i,1],'kx')
    #b = axes2.plot(data_features[i,:],'b')
    #axes2.axis('off')

for i in classesP:
    #axes2 = fig.add_axes([(means[i,0]-(-0.006))/0.016*0.8+0.1-0.01, (means[i,1]-(-0.008))/0.016*0.8+0.1-0.01, 0.05,0.05])
    #axes2.set_alpha(0.0)
    axes1.plot(means[i,0], means[i,1],'kx')
    #p = axes2.plot(data_features[i,:],'g')
    #axes2.axis('off')

for i in classesD:
    #axes2 = fig.add_axes([(means[i,0]-(-0.006))/0.016*0.8+0.1-0.01, (means[i,1]-(-0.008))/0.016*0.8+0.1-0.01, 0.05,0.05])
    #axes2.set_alpha(0.0)
    axes1.plot(means[i,0], means[i,1],'kx')
    #d = axes2.plot(data_features[i,:],'m')
    #axes2.axis('off')
#axes1.set_xlabel('x')
#axes1.set_ylabel('y')
#axes1.set_title('title')

#plt.legend(loc='upper center', shadow=True)


#plt.connect('motion_notify_event', re_zoom)  # for right-click pan/zoom
plt.connect('button_release_event',re_zoom)  # for rectangle-select zoom
#fig.show()
plt.show()