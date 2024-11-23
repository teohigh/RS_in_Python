# load in the data
import os
import pickle
from builtins import range
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

if not os.path.exists('json/user2movie.json') or \
    not os.path.exists('json/movie2user.json') or \
    not os.path.exists('json/usermovie2rating.json') or \
    not os.path.exists('json/usermovie2rating_test.json'):
  pass

with open('json/user2movie.json', 'rb') as f:
    user2movie = pickle.load(f)
with open('json/movie2user.json', 'rb') as f:
    movie2user = pickle.load(f)
with open('json/usermovie2rating.json', 'rb') as f:
    usermovie2rating = pickle.load(f)
with open('json/usermovie2rating_test.json', 'rb') as f:
    usermovie2rating_test = pickle.load(f)

N = np.max(list(user2movie.keys())) + 1
# the test set may contain movies the train set doesn't have data on
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
M = max(m1, m2) + 1
print("N:", N, "M:", M)

K = 10 # Kích thước của không gian tiềm ẩn
W = np.random.randn(N, K) # Ma trận trọng số người dùng với mỗi hàng là vector K-chiều của một người dùng.
b = np.zeros(N) # Bias người dùng
U = np.random.randn(M, K) # Ma trận trọng số phim với mỗi hàng là vector K-chiều của một phim.
c = np.zeros(M) # Bias phim
mu = np.mean(list(usermovie2rating.values())) # Giá trị trung bình xếp hạng

# prediction[i,j] = W[i].dot(U[j]) + b[i] + c.T[j] + mu

# trả về giá trị
# lỗi trung bình bình phương trên toàn bộ tập dữ liệu d.
def get_loss(d):
  N = float(len(d))
  sse = 0 # sum of squared errors
  for k, r in d.items():
    i, j = k
    p = W[i].dot(U[j]) + b[i] + c[j] + mu
    sse += (p - r)*(p - r)
  return sse / N


# Huấn luyện mô hình
epochs = 25
reg =20. # regularization penalty
train_losses = []
test_losses = []
for epoch in range(epochs):
  print("epoch:", epoch)
  epoch_start = datetime.now()

  # update W and b
  t0 = datetime.now()
  for i in range(N):
    # for W
    matrix = np.eye(K) * reg
    vector = np.zeros(K)

    # for b
    bi = 0
    for j in user2movie[i]:
      r = usermovie2rating[(i,j)]
      matrix += np.outer(U[j], U[j])
      vector += (r - b[i] - c[j] - mu)*U[j]
      bi += (r - W[i].dot(U[j]) - c[j] - mu)

    # set the updates
    W[i] = np.linalg.solve(matrix, vector)
    b[i] = bi / (len(user2movie[i]) + reg)

    if i % (N//10) == 0:
      print("i:", i, "N:", N)
  print("updated W and b:", datetime.now() - t0)

  # update U and c
  t0 = datetime.now()
  for j in range(M):
    # for U
    matrix = np.eye(K) * reg
    vector = np.zeros(K)

    # for c
    cj = 0
    try:
      for i in movie2user[j]:
        r = usermovie2rating[(i,j)]
        matrix += np.outer(W[i], W[i])
        vector += (r - b[i] - c[j] - mu)*W[i]
        cj += (r - W[i].dot(U[j]) - b[i] - mu)

      # set the updates
      U[j] = np.linalg.solve(matrix, vector)
      c[j] = cj / (len(movie2user[j]) + reg)

      if j % (M//10) == 0:
        print("j:", j, "M:", M)
    except KeyError:
      # possible not to have any ratings for a movie
      pass
  print("updated U and c:", datetime.now() - t0)
  print("epoch duration:", datetime.now() - epoch_start)

  t0 = datetime.now()
  train_losses.append(get_loss(usermovie2rating))
  test_losses.append(get_loss(usermovie2rating_test))
  print("calculate cost:", datetime.now() - t0)
  print("train loss:", train_losses[-1])
  print("test loss:", test_losses[-1])


print("train losses:", train_losses)
print("test losses:", test_losses)

# plot losses
plt.plot(train_losses, label="train loss")
plt.plot(test_losses, label="test loss")
plt.legend()
plt.show()