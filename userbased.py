from __future__ import print_function, division

import os
import pickle
from builtins import range

import numpy as np
from sortedcontainers import SortedList

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
# Số lượng phim (tính dựa trên cả tập huấn luyện và tập kiểm tra)
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
M = max(m1, m2) + 1
print("N:", N, "M:", M)

if N > 10000:
  print("N =", N, "are you sure you want to continue?")
  print("Comment out these lines if so...")
  exit()

# để tìm các người dùng tương tự phải thực hiện O(N^2 * M) phép tính
# trong thực tế, nên song song hóa việc này (parallelize)
# lưu ý: thực ra chỉ cần thực hiện nửa số phép tính, vì w_ij là đối xứng (symmetric)


# Mỗi người dùng được so sánh với những người dùng khác
# để xác định độ tương đồng dựa trên hệ số tương quan Pearson
K = 25 # có thể giới hạn số K để giảm độ phức tạp
limit = 5 # giới hạn số phim chung giữa 2 người dùng
neighbors = []
averages = []
deviations = []

for i in range(N):
  # tìm K người dùng gần nhất với người dùng i
  movies_i = user2movie[i]
  movies_i_set = set(movies_i)

  # tính trung bình và độ lệch chuẩn
  ratings_i = { movie:usermovie2rating[(i, movie)] for movie in movies_i }
  avg_i = np.mean(list(ratings_i.values()))
  dev_i = { movie:(rating - avg_i) for movie, rating in ratings_i.items() }
  dev_i_values = np.array(list(dev_i.values()))
  sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))

  # lưu trữ trung bình và độ lệch chuẩn
  averages.append(avg_i)
  deviations.append(dev_i)

  sl = SortedList()
  for j in range(N):
    # không so sánh với chính mình
    if j != i:
      movies_j = user2movie[j]
      movies_j_set = set(movies_j)
      common_movies = (movies_i_set & movies_j_set) # intersection (phép giao)
      if len(common_movies) > limit:
        # tính trung bình và độ lệch chuẩn
        ratings_j = { movie:usermovie2rating[(j, movie)] for movie in movies_j }
        avg_j = np.mean(list(ratings_j.values()))
        dev_j = { movie:(rating - avg_j) for movie, rating in ratings_j.items() }
        dev_j_values = np.array(list(dev_j.values()))
        sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))

        # tính hệ số tương quan Pearson
        numerator = sum(dev_i[m]*dev_j[m] for m in common_movies)
        w_ij = numerator / (sigma_i * sigma_j)

        # chèn vào danh sách sắp xếp và cắt bớt
        # negate weight, because list is sorted ascending
        # giảm giá trị của w_ij vì danh sách được sắp xếp tăng dần
        # maximum value (1) is "closest"
        sl.add((-w_ij, j))
        if len(sl) > K:
          del sl[-1]

  # lưu trữ K người dùng gần nhất
  neighbors.append(sl)

  # in ra số lượng người dùng đã xử lý
  if i % 1 == 0:
    print(i)


# Dự đoán = Trung bình trọng số của xếp hạng từ các hàng xóm,
# cộng với trung bình đánh giá của người dùng mục tiêu.
def predict(i, m):
  # tính trọng số độ lệch
  numerator = 0 # tử số: tính tổng trọng số các độ lệch
  denominator = 0 # mẫu số: tổng trọng số tuyệt đối (để chuẩn hóa)
  for neg_w, j in neighbors[i]:
    # lưu ý, trọng số được lưu dưới dạng số âm
    # vì vậy, trọng số âm của trọng số âm là trọng số dương
    try:
      numerator += -neg_w * deviations[j][m]
      denominator += abs(neg_w)
    except KeyError:
      # hàng xóm có thể chưa đánh giá cùng bộ phim
      # không thực hiện tra cứu từ điển hai lần
      # chỉ cần bỏ qua
      pass

  if denominator == 0:
    prediction = averages[i]
  else:
    prediction = numerator / denominator + averages[i]
  prediction = min(5, prediction)
  prediction = max(0.5, prediction) # min rating is 0.5
  return prediction

# tính dự đoán trên tập huấn luyện và tập kiểm tra
train_predictions = []
train_targets = []
for (i, m), target in usermovie2rating.items():
  prediction = predict(i, m)
  train_predictions.append(prediction)
  train_targets.append(target)

test_predictions = []
test_targets = []
for (i, m), target in usermovie2rating_test.items():
  prediction = predict(i, m)
  test_predictions.append(prediction)
  test_targets.append(target)


# tính toán lỗi bình phương trung bình (Mean Squared Error - MSE)
# trên cả tập huấn luyện và kiểm tra
def mse(p, t):
  p = np.array(p)
  t = np.array(t)
  return np.mean((p - t)**2)

print('train mse:', mse(train_predictions, train_targets))
print('test mse:', mse(test_predictions, test_targets))