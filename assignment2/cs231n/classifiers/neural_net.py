from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################

    # FC1 layer.
    fc1_activation = np.dot(X, W1) + b1

    # Relu layer.
    relu_1_activation = fc1_activation
    relu_1_activation[relu_1_activation < 0] = 0

    # FC2 layer.
    #fc2_activation = np.dot(relu_1_activation, W2) + b2
    # 문제 1-1: 위 구문(line: 88)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
    fc2_activation = np.zeros ([relu_1_activation.shape[0], W2.shape[1]])
    for i in range (relu_1_activation.shape[0]):
      for j in range (W2.shape[1]):
        for k in range (relu_1_activation.shape[1]):
          fc2_activation[i,j] += relu_1_activation[i,k] * W2[k,j]
        fc2_activation[i,j] += b2[j]


    # Output scores.
    scores = fc2_activation

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = 0.0
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################

    shift_scores = scores - np.max(scores, axis=1)[..., np.newaxis]

    softmax_scores = np.exp(shift_scores) / np.sum(np.exp(shift_scores), axis=1)[..., np.newaxis]

    correct_class_scores = np.choose(y, shift_scores.T) # Size N vector
    #loss = -correct_class_scores + np.log(np.sum(np.exp(shift_scores), axis=1))
    #loss = np.sum(loss)

    #loss /= N
    #loss += reg * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(b1*b1) + np.sum(b2*b2))
    # 문제 2: 위 구문(line: 117, 118, 121)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
    
    temp = np.zeros ([shift_scores.shape[0]])
    for i in range (shift_scores.shape[0]):
      for j in range (shift_scores.shape[1]):
        temp[i] += np.exp(shift_scores)[i,j]
    temp2 = -correct_class_scores + np.log (temp)
    loss = 0
    for i in range (temp2.shape[0]):
      loss += temp2[i]
    
    loss /= N
    tmp = 0
    for i in range (W1.shape[0]):
      for j in range (W1.shape[1]):
        tmp += W1[i,j] * W1[i,j]
    for i in range (W2.shape[0]):
      for j in range (W2.shape[1]):
        tmp += W2[i,j] * W2[i,j]
    for i in range (b1.shape[0]):
      tmp += b1[i] * b1[i]
    for i in range (b2.shape[0]):
      tmp += b2[i] * b2[i]
    loss += reg * tmp
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################

    dSoft = softmax_scores
    dSoft[range(N),y] = dSoft[range(N),y] - 1
    dSoft /= N  # Average over batch.

    #dW2 = np.dot(relu_1_activation.T, dSoft)
    # 문제 3-1: 위 구문(line: 140)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
    dW2 = np.zeros ([relu_1_activation.T.shape[0], dSoft.shape[1]])
    for i in range (relu_1_activation.T.shape[0]):
      for j in range (dSoft.shape[1]):
        for k in range (dSoft.shape[0]):
          dW2[i,j] += relu_1_activation.T[i,k] * dSoft[k,j]
    
    dW2 += 2*reg*W2
    grads['W2'] = dW2
    
    db2 = dSoft * 1
    #grads['b2'] = np.sum(db2, axis=0)
    # 문제 3-2: 위 구문(line: 146)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
    tmp = np.zeros ([db2.shape[1]])
    for j in range (db2.shape[1]):
      for i in range (db2.shape[0]):
        tmp[j] += db2[i,j]
    grads['b2'] = tmp

    #dx2 = np.dot(dSoft, W2.T)
    # 문제 3-3: 위 구문(line: 149)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
    dx2 = np.zeros ([dSoft.shape[0], W2.T.shape[1]])
    for i in range (dSoft.shape[0]):
      for j in range (W2.T.shape[1]):
        for k in range (W2.T.shape[0]):
          dx2[i,j] += dSoft[i][k] * W2.T[k][j]

    relu_mask = (relu_1_activation > 0)
    dRelu1= relu_mask*dx2

    #dW1 = np.dot(X.T, dRelu1)
    # 문제 3-4: 위 구문(line: 154)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
    dW1 = np.zeros ([X.T.shape[0], dRelu1.shape[1]])
    for i in range (X.T.shape[0]):
      for j in range (dRelu1.shape[1]):
        for k in range (dRelu1.shape[0]):
          dW1[i,j] += X.T[i][k] * dRelu1[k][j]

    dW1 += 2*reg*W1
    grads['W1'] = dW1

    db1 = dRelu1 * 1
    #grads['b1'] = np.sum(db1, axis=0)
    # 문제 3-5: 위 구문(line: 160)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
    tmp7 = np.zeros ([db1.shape[1]])
    for j in range (db1.shape[1]):
      for i in range (db1.shape[0]):
        tmp7[j] += db1[i,j]
    grads['b1'] = tmp7

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################

      random_batch = np.random.permutation(num_train)[0:batch_size]
      X_batch = X[random_batch,...]
      y_batch = y[random_batch]
        
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)
        
      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################

      self.params['W1'] += -grads['W1']*learning_rate
      self.params['b1'] += -grads['b1']*learning_rate
      self.params['W2'] += -grads['W2']*learning_rate
      self.params['b2'] += -grads['b2']*learning_rate
    
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################

    #y_pred = np.argmax(self.loss(X), axis=1)
    # 문제 4: 위 구문(line: 275)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
    y_pred = np.zeros ([self.loss(X).shape[0]], dtype=int)
    for l in range (self.loss(X).shape[0]):
      max_val = -10000
      max_idx = 0
      for m in range (self.loss(X).shape[1]):
        if max_val < self.loss(X)[l][m]:
          max_val = self.loss(X)[l][m]
          max_idx = m
      y_pred[l] = max_idx

    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


