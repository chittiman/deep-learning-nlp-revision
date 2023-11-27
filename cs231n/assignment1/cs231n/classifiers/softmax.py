from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    D,C = W.shape
    N = y.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i in range(N):
        X_i,y_i = X[i,:],y[i] # (1,D) , (1,)
        scores_i = X_i@W # (1,C)
        scores_i -= scores_i.max() #(1,C)
        exp_scores_i = np.exp(scores_i) #(1,C)
        exp_norm_i = exp_scores_i / exp_scores_i.sum() #(1,C)
        loss_i = -np.log(exp_norm_i[y_i])
        loss += loss_i

        exp_norm_i[y_i] -=  1 #(1,C)
        dW_i = X_i.reshape(-1,1) @ exp_norm_i.reshape(1,-1) #(D,1)@(1,C)
        dW += dW_i  


    loss /= N
    loss += reg * np.sum(W * W)

    dW /= N
    dW += 2*reg*W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_dims = W.shape[0]
    num_classes = W.shape[1]
    num_train = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X@W #(N,C)
    scores -= scores.max(axis=-1).reshape(-1,1)
    scores_exp = np.exp(scores)
    scores_norm = scores_exp / scores_exp.sum(axis=-1).reshape(-1,1)
    scores_labels = scores_norm[np.arange(num_train),y]
    loss_array = np.log(scores_labels)

    loss = -loss_array.mean()
    loss += reg * np.sum(W * W)    

    d_scores = scores_norm #(N,C)
    d_scores[np.arange(num_train),y] -= 1
    dW =  X.T @ d_scores#(D,C) = (D,N) @(N,C)

    dW /= num_train
    dW += 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
