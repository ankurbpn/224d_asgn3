import numpy as np
import collections
np.seterr(over='raise',under='raise')

def tanh(x):
	return (2/(1 + np.exp(-2*x))) - np.ones(x.shape)

class RNTN:

    def __init__(self,wvecDim,outputDim,numWords,mbSize=30,rho=1e-4):
        self.wvecDim = wvecDim
        self.outputDim = outputDim
        self.numWords = numWords
        self.mbSize = mbSize
        self.defaultVec = lambda : np.zeros((wvecDim,))
        self.rho = rho
	self.p = 1.0
	self.implementDropout = False

    def initParams(self):
        np.random.seed(12341)
        
	if self.implementDropout:
		self.p = 0.5

        # Word vectors
        self.L = 0.01*np.random.randn(self.wvecDim,self.numWords)

        # Hidden activation weights
        self.V = 0.01*np.random.randn(self.wvecDim,2*self.wvecDim,2*self.wvecDim)
        self.W = 0.01*np.random.randn(self.wvecDim,self.wvecDim*2)
        self.b = np.zeros((self.wvecDim))

        # Softmax weights
        self.Ws = 0.01*np.random.randn(self.outputDim,self.wvecDim)
        self.bs = np.zeros((self.outputDim))

        self.stack = [self.L, self.V, self.W, self.b, self.Ws, self.bs]

        # Gradients
        self.dV = np.empty((self.wvecDim,2*self.wvecDim,2*self.wvecDim))
        self.dW = np.empty(self.W.shape)
        self.db = np.empty((self.wvecDim))
        self.dWs = np.empty(self.Ws.shape)
        self.dbs = np.empty((self.outputDim))

    def costAndGrad(self,mbdata,test=False): 
        cost = 0.0
        correct = []
        guess = []
        total = 0.0
	
	self.L, self.V, self.W, self.b, self.Ws, self.bs = self.stack
        # Zero gradients
        self.dV[:] = 0
        self.dW[:] = 0
        self.db[:] = 0

        self.dWs[:] = 0
        self.dbs[:] = 0
        self.dL = collections.defaultdict(self.defaultVec)


        # Forward prop each tree in minibatch
        for tree in mbdata: 
            c,tot = self.forwardProp(tree.root,correct,guess, test)
            cost += c
            total += tot
	   
        if test:
            return (1./len(mbdata))*cost,correct, guess, total

        # Back prop each tree in minibatch
        for tree in mbdata:
            self.backProp(tree.root)

        # scale cost and grad by mb size
        scale = (1./self.mbSize)
        for v in self.dL.itervalues():
            v *=scale
        
        # Add L2 Regularization 
        cost += (self.rho/2)*np.sum(self.V**2)
        cost += (self.rho/2)*np.sum(self.W**2)
        cost += (self.rho/2)*np.sum(self.Ws**2)
	print self.implementDropout
        return scale*cost,[self.dL,scale*(self.dV + self.rho*self.V),
                                   scale*(self.dW + self.rho*self.W),scale*self.db,
                                   scale*(self.dWs+self.rho*self.Ws),scale*self.dbs]


    def forwardProp(self,node, correct = [], guess= [], test = False):
        cost = total = 0.0
	if node.isLeaf:
		self.L[:, node.word] = self.L[:, node.word]/np.sqrt(np.sum(self.L[:, node.word]**2))
		node.hActs1 = self.L[:, node.word]
		#node.hActs1[node.hActs1<0] = 0

		if self.implementDropout and not test:
			#We use node.hActs2 to store the mask
			node.hActs2 = np.random.binomial(1,self.p, node.hActs1.shape)
			node.hActs1 = np.multiply(node.hActs2, node.hActs1)
			
		if self.implementDropout and test:
			node.hActs1 = self.p*node.hActs1

		node.probs = np.dot(self.Ws, node.hActs1) + self.bs
		node.probs = np.exp(node.probs - np.max(node.probs))
		node.probs = node.probs/np.sum(node.probs)
		correct.append(node.label)
		guess.append(np.argmax(node.probs))
		cost = -np.log(node.probs[node.label])
		#total -= 1
        else:
		retl = self.forwardProp(node.left, correct, guess, test)
		retr = self.forwardProp(node.right, correct, guess, test)
		cost = retl[0] + retr[0]
		total = retl[1] + retr[1]
		node.hActs1 = np.dot(self.W, np.concatenate((node.left.hActs1, node.right.hActs1))) + self.b + np.dot(np.dot(self.V, np.concatenate((node.left.hActs1, node.right.hActs1))),np.concatenate((node.left.hActs1, node.right.hActs1)))

		if self.implementDropout and not test:
			node.hActs2 = np.random.binomial(1,self.p, node.hActs1.shape)
			node.hActs1 = np.multiply(node.hActs1, node.hActs2)

		if self.implementDropout and test:
			node.hActs1 = self.p*node.hActs1

		node.hActs1 = tanh(node.hActs1)
		node.probs = np.dot(self.Ws, node.hActs1) + self.bs

		node.probs = np.exp(node.probs - np.max(node.probs))
		node.probs = node.probs/np.sum(node.probs)
		correct.append(node.label)
		guess.append(np.argmax(node.probs))
		cost += -np.log(node.probs[node.label])
        
        return cost,total + 1


    def backProp(self,node,error=None):

        # Clear nodes
        node.fprop = False
	#this is exactly the same setup as backProp in rnn.py
        del0 = node.probs
	del0[node.label] -= 1
	self.dWs += np.outer(del0, node.hActs1)
	self.dbs += del0
	del1 = np.dot(np.transpose(self.Ws), del0)

	if error is not None:
		del1 += error

	if self.implementDropout:
		del1 = np.multiply(del1, node.hActs2)

	if node.isLeaf:
		#sgns = np.zeros(del1.shape)
		#sgns[node.hActs1 > 0] = 1
		self.dL[node.word] += del1

	else:
		del1 = np.multiply(del1, np.ones(node.hActs1.shape) - node.hActs1**2)
		self.db += del1
		hActschild = np.concatenate((node.left.hActs1, node.right.hActs1))
		self.dW += np.outer(del1, hActschild)
		vs = (del1, hActschild, hActschild)
		self.dV += reduce(np.multiply, np.ix_(*vs))
		delbelow = np.dot(np.transpose(self.W), del1) + np.dot(np.dot(np.transpose(self.V), del1), hActschild)
		self.backProp(node.left, delbelow[:self.wvecDim])
		self.backProp(node.right, delbelow[self.wvecDim:])
        
    def updateParams(self,scale,update,log=False):
        """
        Updates parameters as
        p := p - scale * update.
        If log is true, prints root mean square of parameter
        and update.
        """
        if log:
            for P,dP in zip(self.stack[1:],update[1:]):
                pRMS = np.sqrt(np.mean(P**2))
                dpRMS = np.sqrt(np.mean((scale*dP)**2))
                print "weight rms=%f -- update rms=%f"%(pRMS,dpRMS)

        self.stack[1:] = [P+scale*dP for P,dP in zip(self.stack[1:],update[1:])]

        # handle dictionary update sparsely
        dL = update[0]
        for j in dL.iterkeys():
            self.L[:,j] += scale*dL[j]

    def toFile(self,fid):
        import cPickle as pickle
        pickle.dump(self.stack,fid)

    def fromFile(self,fid):
        import cPickle as pickle
        self.stack = pickle.load(fid)

    def check_grad(self,data,epsilon=1e-6):

        cost, grad = self.costAndGrad(data)
        err1 = 0.0
        count = 0.0

        print "Checking dW... (might take a while)"
        for W,dW in zip(self.stack[1:],grad[1:]):
            W = W[...,None,None] # add dimension since bias is flat
            dW = dW[...,None,None] 
            for i in xrange(W.shape[0]):
                for j in xrange(W.shape[1]):
                    for k in xrange(W.shape[2]):
                        W[i,j,k] += epsilon
                        costP,_ = self.costAndGrad(data)
                        W[i,j,k] -= epsilon
                        numGrad = (costP - cost)/epsilon
                        err = np.abs(dW[i,j,k] - numGrad)
                        #print "Analytic %.9f, Numerical %.9f, Relative Error %.9f"%(dW[i,j,k],numGrad,err)
                        err1+=err
                        count+=1
        if 0.001 > err1/count:
            print "Grad Check Passed for dW"
        else:
            print "Grad Check Failed for dW: Sum of Error = %.9f" % (err1/count)

        # check dL separately since dict
        dL = grad[0]
        L = self.stack[0]
        err2 = 0.0
        count = 0.0
        print "Checking dL..."
        for j in dL.iterkeys():
            for i in xrange(L.shape[0]):
                L[i,j] += epsilon
                costP,_ = self.costAndGrad(data)
                L[i,j] -= epsilon
                numGrad = (costP - cost)/epsilon
                err = np.abs(dL[j][i] - numGrad)
                #print "Analytic %.9f, Numerical %.9f, Relative Error %.9f"%(dL[j][i],numGrad,err)
                err2+=err
                count+=1

        if 0.001 > err2/count:
            print "Grad Check Passed for dL"
        else:
            print "Grad Check Failed for dL: Sum of Error = %.9f" % (err2/count)

if __name__ == '__main__':

    import tree as treeM
    train = treeM.loadTrees()
    numW = len(treeM.loadWordMap())

    wvecDim = 10
    outputDim = 5

    nn = RNTN(wvecDim,outputDim,numW,mbSize=4)
    nn.initParams()

    mbData = train[:1]
    #cost, grad = nn.costAndGrad(mbData)

    print "Numerical gradient check..."
    nn.check_grad(mbData)






