import numpy as np
from random import shuffle

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_classes = W.shape[1]
  num_train = X.shape[0]

  scores = np.dot(X, W)
  # 문제 6-1: 위 구문(line:36)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
  '''
  scores = np.zeros ([X.shape[0], W.shape[1]])
  for i in range (X.shape[0]):
    for j in range (W.shape[1]):
      for k in range (X.shape[1]):
        scores[i,j] = scores[i,j] + X[i,k] * W[k,j]
  '''

  for ii in range(num_train):
    current_scores = scores[ii, :]

    shift_scores = current_scores - np.max(current_scores)

    loss_ii = -shift_scores[y[ii]] + np.log(np.sum(np.exp(shift_scores)))
    # 문제 6-2: 위 구문(line:43)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
    '''
    tmp = 0.0
    for k in range (shift_scores.shape[0]):
      tmp = tmp + np.exp (shift_scores[k])
    tmp = np.log (tmp)
    loss_ii = -shift_scores[y[ii]] + tmp
    '''

    loss += loss_ii

    for jj in range(num_classes):
      softmax_score = np.exp(shift_scores[jj]) / np.sum(np.exp(shift_scores))
      # 문제 6-3: 위 구문(line:43)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
      '''
      tmp2 = 0.0
      for l in range (shift_scores.shape[0]):
        tmp2 = tmp2 + np.exp (shift_scores[l])
      softmax_score = np.exp (shift_scores[jj]) / tmp2
      '''

      if jj == y[ii]:
        dW[:, jj] += (-1 + softmax_score) * X[ii]
      else:
        dW[:, jj] += softmax_score * X[ii]

  loss /= num_train
  loss += reg * np.sum(W*W)
  # 문제 6-4: 위 구문(line:43)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
  '''
  tmp3 = 0.0
  for m in range (W.shape[0]):
    for n in range (W.shape[1]):
      tmp3 = tmp3 + W[m,n] * W[m,n]
  loss += reg * tmp3
  '''


  dW /= num_train
  dW += 2*reg*W

  # (list to np array, tuple to np array 변환 함수(np.array())는 사용 가능)

    
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_train = X.shape[0]

  scores = np.dot(X, W)
  shift_scores = scores - np.max(scores, axis=1)[...,np.newaxis]
  # 문제 7-1: 위 구문(line: 96)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
  '''
  tmp = np.zeros ([scores.shape[0],1])
  for i in range (scores.shape[0]):
    max_val = 0
    for j in range (scores.shape[1]):
      if max_val < scores[i,j]:
        max_val = scores[i,j]
    tmp[i,0] = max_val
  shift_scores = scores - tmp
  '''

  softmax_scores = np.exp(shift_scores)/ np.sum(np.exp(shift_scores), axis=1)[..., np.newaxis]
  # 문제 7-2: 위 구문(line: 99)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
  '''
  tmp = np.zeros ([shift_scores.shape[0], 1])
  for i in range (shift_scores.shape[0]):
    for j in range (shift_scores.shape[1]):
      tmp[i,0] += np.exp (shift_scores[i,j])
  softmax_scores = np.exp(shift_scores) / tmp  
  '''

  dScore = softmax_scores
  dScore[range(num_train),y] = dScore[range(num_train),y] - 1

  dW = np.dot(X.T, dScore)
  dW /= num_train
  dW += 2*reg*W

  correct_class_scores = np.choose(y, shift_scores.T)  # Size N vector
  loss = -correct_class_scores + np.log(np.sum(np.exp(shift_scores), axis=1))
  loss = np.sum(loss)
  # 문제 7-3: 위 구문(line: 110, 111)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
  '''
  tmp2 = np.zeros ([shift_scores.shape[0]])
  for i in range (shift_scores.shape[0]):
    t_sum = 0.0
    for j in range (shift_scores.shape[1]):
      t_sum += np.exp (shift_scores[i,j])
    tmp2[i] = np.log (t_sum)
  tmp3 = -correct_class_scores + tmp2
  loss = 0.0
  for k in range (tmp3.shape[0]):
    loss += tmp3[k]
  '''
  

  loss /= num_train
  loss += reg * np.sum(W*W)
  # 문제 7-4: 위 구문(line: 110, 111)을 numpy lib를 사용하지 않고 numpy lib를 사용한 결과와 동일하게 동작하도록 작성
  '''
  loss /= num_train
  tmp4 = 0.0
  for m in range (W.shape[0]):
    for n in range (W.shape[1]):
      tmp4 += W[m,n] * W[m,n]
  loss += reg * tmp4
  '''

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

