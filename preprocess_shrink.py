import pandas as pd
from collections import Counter
#  thu nhỏ dữ liệu từ file edited_rating.csv bằng cách
#  giữ lại một số lượng người dùng và phim nhất định,
#  sau đó lưu dữ liệu đã thu nhỏ vào file small_rating.csv

df = pd.read_csv('archive/edited_rating.csv')
print("original dataframe size: ", len(df))

N = df.userId.max() + 1
M = df.movie_idx.max() + 1

# Đếm số lượng đánh giá của từng người dùng và phim
user_ids_count = Counter(df.userId)
movie_ids_count = Counter(df.movie_idx)

# Số lượng người dùng và phim muốn giữ lại
n = 1000
m = 200

# Lấy ra n người dùng và m phim có số lượng đánh giá nhiều nhất
user_ids = [u for u, c in user_ids_count.most_common(n)]
movie_ids = [m for m, c in movie_ids_count.most_common(m)]

# Tạo dataframe mới chỉ chứa các dòng của n người dùng và m phim vừa chọn
df_small = df[df.userId.isin(user_ids) & df.movie_idx.isin(movie_ids)].copy()

# Tạo lại các ID người dùng và phim vì chúng không còn liên tiếp
new_user_id_map = {}
i = 0
for old in user_ids:
    new_user_id_map[old] = i
    i += 1
print("i:", i)

new_movie_id_map = {}
j = 0
for old in movie_ids:
    new_movie_id_map[old] = j
    j += 1
print("j:", j)

# Cập nhật lại các ID người dùng và phim trong dataframe mới
print("Setting new ids")
df_small.loc[:, 'userId'] = df_small.apply(lambda row: new_user_id_map[row.userId], axis=1)
df_small.loc[:, 'movie_idx'] = df_small.apply(lambda row: new_movie_id_map[row.movie_idx], axis=1)

print("max user id:", df_small.userId.max())
print("max movie id:", df_small.movie_idx.max())
print("small dataframe size:", len(df_small))
print("Saving small dataframe")
df_small.to_csv('archive/small_rating.csv', index=False)