import numpy as np
import collections
import pdb



class RNN3:

    def __init__(self,wvecDim, middleDim, outputDim,numWords,mbSize=30,rho=1e-4):
        self.wvecDim = wvecDim
        self.outputDim = outputDim
        self.middleDim = middleDim
        self.numWords = numWords
        self.mbSize = mbSize
        self.defaultVec = lambda : np.zeros((wvecDim,))
        self.rho = rho

    def initParams(self):
        np.random.seed(12341)

        # Word vectors
        self.L = 0.01*np.random.randn(self.wvecDim,self.numWords)

        # Hidden activation weights for layer 1
        self.W1 = 0.01*np.random.randn(self.wvecDim,2*self.wvecDim)
        self.b1 = np.zeros((self.wvecDim))

        # Hidden activation weights for layer 2
        self.W2 = 0.01*np.random.randn(self.middleDim,self.wvecDim)
        self.b2 = np.zeros((self.middleDim))

        # Softmax weights
        self.Ws = 0.01*np.random.randn(self.outputDim,self.middleDim) # note this is " U " in the notes and the handout.. there is a reason for the change in notation
        self.bs = np.zeros((self.outputDim))
	
	self.Ws_nl =  0.01*np.random.randn(self.outputDim,self.middleDim)
	self.bs_nl = np.zeros((self.outputDim))

        self.stack = [self.L, self.W1, self.b1, self.W2, self.b2, self.Ws, self.bs, self.Ws_nl, self.bs_nl]

        # Gradients
        self.dW1 = np.empty(self.W1.shape)
        self.db1 = np.empty((self.wvecDim))
        
        self.dW2 = np.empty(self.W2.shape)
        self.db2 = np.empty((self.middleDim))

        self.dWs = np.empty(self.Ws.shape)
        self.dbs = np.empty((self.outputDim))
        self.dWs_nl = np.empty(self.Ws_nl.shape)
        self.dbs_nl = np.empty((self.outputDim))

    def costAndGrad(self,mbdata,test=False): 
        """
        Each datum in the minibatch is a tree.
        Forward prop each tree.
        Backprop each tree.
        Returns
           cost
           Gradient w.r.t. W1, W2, Ws, b1, b2, bs
           Gradient w.r.t. L in sparse form.

        or if in test mode
        Returns 
           cost, correctArray, guessArray, total
        """
        cost = 0.0
        correct = []
        guess = []
        total = 0.0

        self.L, self.W1, self.b1, self.W2, self.b2, self.Ws, self.bs, self.Ws_nl, self.bs_nl = self.stack
        # Zero gradients
        self.dW1[:] = 0
        self.db1[:] = 0
        
        self.dW2[:] = 0
        self.db2[:] = 0

        self.dWs[:] = 0
        self.dbs[:] = 0
        self.dL = collections.defaultdict(self.defaultVec)
        self.dWs_nl[:] = 0
        self.dbs_nl[:] = 0

        # Forward prop each tree in minibatch
        for tree in mbdata: 
            c,tot = self.forwardProp(tree.root,correct,guess)
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
        cost += (self.rho/2)*np.sum(self.W1**2)
        cost += (self.rho/2)*np.sum(self.W2**2)
        cost += (self.rho/2)*np.sum(self.Ws**2)
	cost += (self.rho/2)*np.sum(self.Ws_nl**2)

        return scale*cost,[self.dL,scale*(self.dW1 + self.rho*self.W1),scale*self.db1,
                                   scale*(self.dW2 + self.rho*self.W2),scale*self.db2,
                                   scale*(self.dWs+self.rho*self.Ws),scale*self.dbs,
				   scale*(self.dWs_nl + self.rho*self.Ws_nl),scale*self.dbs_nl]


    def forwardProp(self,node, correct=[], guess=[]):
        cost  =  total = 0.0
        # this is exactly the same setup as forwardProp in rnn.py
        if node.isLeaf:
		node.hActs1 = self.L[:, node.word]
		#node.hActs1[node.hActs1<0] = 0
		node.hActs2 =  np.dot(self.W2, node.hActs1) + self.b2
		node.hActs2[node.hActs2 <0] = 0
		node.probs = np.dot(self.Ws, node.hActs2) + self.bs
		node.probs = np.exp(node.probs - np.max(node.probs))
		node.probs = node.probs/np.sum(node.probs)
		#correct.append(node.label)
		#guess.append(np.argmax(node.probs))
		cost = -np.log(node.probs[node.label])
		total -= 1
        else:
		retl = self.forwardProp(node.left, correct, guess)
		retr = self.forwardProp(node.right, correct, guess)
		cost = retl[0] + retr[0]
		total = retl[1] + retr[1]
		node.hActs1 = np.dot(self.W1, np.concatenate((node.left.hActs1, node.right.hActs1))) + self.b1
		node.hActs1[node.hActs1 <0] = 0
		node.hActs2 =  np.dot(self.W2, node.hActs1) + self.b2
		node.hActs2[node.hActs2 <0] = 0
		node.probs = np.dot(self.Ws_nl, node.hActs2) + self.bs_nl
		node.probs = np.exp(node.probs - np.max(node.probs))
		node.probs = node.probs/np.sum(node.probs)
		correct.append(node.label)
		guess.append(np.argmax(node.probs))
		cost += -np.log(node.probs[node.label])
	return cost, total + 1

    def backProp(self,node,error=None):

        # this is exactly the same setup as backProp in rnn.py
	if node.isLeaf:
		del0 = node.probs
		del0[node.label] -= 1
		self.dWs += np.outer(del0, node.hActs2)
		self.dbs += del0
		del1 = np.dot(np.transpose(self.Ws), del0)
		sgns = np.zeros(del1.shape)
		sgns[node.hActs2 > 0] = 1
		del1 = np.multiply(del1, sgns)
		self.db2 += del1
		self.dW2 += np.outer(del1, node.hActs1)
		del2 = np.dot(np.transpose(self.W2), del1)

		if error is not None:
			del2 += error
		#sgns = np.zeros(del1.shape)
		#sgns[node.hActs1 > 0] = 1
		self.dL[node.word] += del2

	else:
		del0 = node.probs
		del0[node.label] -= 1
		self.dWs_nl += np.outer(del0, node.hActs2)
		self.dbs_nl += del0
		del1 = np.dot(np.transpose(self.Ws_nl), del0)
		sgns = np.zeros(del1.shape)
		sgns[node.hActs2 > 0] = 1
		del1 = np.multiply(del1, sgns)
		self.db2 += del1
		self.dW2 += np.outer(del1, node.hActs1)
		del2 = np.dot(np.transpose(self.W2), del1)

		if error is not None:
			del2 += error
		sgns = np.zeros(del2.shape)
		sgns[node.hActs1 > 0] = 1
		del2 = np.multiply(del2, sgns)
		self.db1 += del2
		self.dW1 += np.outer(del2, np.concatenate((node.left.hActs1, node.right.hActs1)))
		delbelow = np.dot(np.transpose(self.W1), del2)
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

    def check_grad(self,data,epsilon=1e-8):

        cost, grad = self.costAndGrad(data)

        err1 = 0.0
        count = 0.0
        print "Checking dWs, dW1 and dW2..."
        for W,dW in zip(self.stack[1:],grad[1:]):
            W = W[...,None] # add dimension since bias is flat
            dW = dW[...,None] 
            for i in xrange(W.shape[0]):
                for j in xrange(W.shape[1]):
                    W[i,j] += epsilon
                    costP,_ = self.costAndGrad(data)
                    W[i,j] -= epsilon
                    numGrad = (costP - cost)/epsilon
                    err = np.abs(dW[i,j] - numGrad)
                    err1+=err
                    count+=1
	    #print dW.shape, err1
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
    middleDim = 10
    outputDim = 5

    rnn = RNN3(wvecDim,middleDim,outputDim,numW,mbSize=4)
    rnn.initParams()

    mbData = train[:4]
    
    print "Numerical gradient check..."
    rnn.check_grad(mbData)






