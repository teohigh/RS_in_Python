from builtins import range

import pickle
import numpy as np
from sortedcontainers import SortedList
import os

from userbased import neighbors

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
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
M = max(m1, m2) + 1
print("Item: N:", N, "M:", M)

if N > 20000:
    print("N =", N, "are you sure you want to continue?")
    print("Comment out these lines if so...")
    exit()

K = 20  # Số lượng hàng xóm cần xét
limit = 5  # Số lượng người dùng chung tối thiểu để xét hai bộ phim là hàng xóm
neighbors = []
averages = []
deviations = []

for i in range(M):
    users_i = movie2user[i] # danh sách người dùng đã xem bộ phim i
    users_i_set = set(users_i)

    # tính hệ số tương quan Pearson cho phim i
    ratings_i = {user:usermovie2rating[(user, i)] for user in users_i}
    avg_i = np.mean(list(ratings_i.values()))
    dev_i = {user:(rating - avg_i) for user, rating in ratings_i.items()}
    dev_i_values = np.array(list(dev_i.values()))
    # Chuẩn hóa độ lệch bằng cách tính căn bậc hai tổng bình phương của các độ lệch
    sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))

    averages.append(avg_i)
    deviations.append(dev_i)

    sl = SortedList()
    for j in range(M):
        if j != i:
            users_j = movie2user[j]
            users_j_set = set(users_j)
            common_users = (users_i_set & users_j_set)
            if len(common_users) > limit:
                # tính hệ số tương quan Pearson cho phim j
                ratings_j = {user:usermovie2rating[(user, j)] for user in users_j}
                avg_j = np.mean(list(ratings_j.values()))
                dev_j = {user:(rating - avg_j) for user, rating in ratings_j.items()}
                dev_j_values = np.array(list(dev_j.values()))
                sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))

                numerator = sum(dev_i[m]*dev_j[m] for m in common_users)
                w_ij = numerator / (sigma_i * sigma_j)

                sl.add((-w_ij, j))
                if len(sl) > K:
                    del sl[-1]

    neighbors.append(sl)

    if i % 1 == 0:
        print(i)

# Dự đoán = Trung bình trọng số của xếp hạng người dùng cho các hàng xóm mục,
# cộng với trung bình đánh giá của mục mục tiêu.
def predict(i, u):
    numerator = 0
    denominator = 0
    for neg_w, j in neighbors[i]:
        try:
            numerator += -neg_w * deviations[j][u]
            denominator += abs(neg_w)
        except KeyError:
            pass

    if denominator == 0:
        prediction = averages[i]
    else:
        prediction = numerator / denominator + averages[i]
    prediction = min(5, prediction)
    prediction = max(0.5, prediction) # min rating is 0.5
    return prediction

train_predictions = []
train_targets = []
for (u, m), target in usermovie2rating.items():
    prediction = predict(m, u)
    train_predictions.append(prediction)
    train_targets.append(target)

test_predictions = []
test_targets = []
for (u, m), target in usermovie2rating_test.items():
    prediction = predict(m, u)
    test_predictions.append(prediction)
    test_targets.append(target)

# MSE (Mean Squared Error): Được sử dụng để đo lường độ chính xác của mô hình: giá trị càng nhỏ càng tốt
def mse(p, t):
    p = np.array(p)
    t = np.array(t)
    return np.mean((p - t)**2)

print("train mse:", mse(train_predictions, train_targets))
print("test mse:", mse(test_predictions, test_targets))
