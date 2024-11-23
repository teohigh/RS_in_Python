import json
import pandas as pd
from sklearn.utils import shuffle
# Đọc dữ liệu từ file small_rating.csv
# và chuyển dữ liệu thành dạng dictionary, sau đó lưu vào file json

df = pd.read_csv('archive/small_rating.csv')
N = df.userId.max() + 1
M = df.movie_idx.max() + 1

df = shuffle(df)  # xáo trộn dữ liệu

# dùng 80% dữ liệu cho training, 20% cho testing
cutoff = int(0.8 * len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

user2movie = {}
movie2user = {}
usermovie2rating = {}

print("Calling: update_user2movie_and_movie2user")
count = 0


def update_user2movie_and_movie2user(row):
    global count
    count += 1
    if count % 100000 == 0:
        print("processed: %.3f" % (float(count) / cutoff))

    i = int(row.userId)
    j = int(row.movie_idx)
    if i not in user2movie:
        user2movie[i] = [j]
    else:
        user2movie[i].append(j)

    if j not in movie2user:
        movie2user[j] = [i]
    else:
        movie2user[j].append(i)
    usermovie2rating[(i, j)] = row.rating


df_train.apply(update_user2movie_and_movie2user, axis=1)

# test ratings dictionary
usermovie2rating_test = {}
print("Calling: update_usermovie2rating_test")
count = 0


def update_usermovie2rating_test(row):
    global count
    count += 1
    if count % 100000 == 0:
        print("processed: %.3f" % (float(count) / len(df_test)))

    i = int(row.userId)
    j = int(row.movie_idx)
    usermovie2rating_test[(i, j)] = row.rating


df_test.apply(update_usermovie2rating_test, axis=1)

# Lưu dữ liệu vào file json
with open('json/user2movie.json', 'w') as f:
    json.dump(user2movie, f)
with open('json/movie2user.json', 'w') as f:
    json.dump(movie2user, f)
with open('json/usermovie2rating.json', 'w') as f:
    json.dump({str(k): v for k, v in usermovie2rating.items()}, f)
with open('json/usermovie2rating_test.json', 'w') as f:
    json.dump({str(k): v for k, v in usermovie2rating_test.items()}, f)
