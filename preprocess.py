import pandas as pd
# Xử lý dữ liệu ban đầu từ file rating.csv,
# chuyển đổi ID người dùng từ 1-based sang 0-based,
# ánh xạ ID phim sang chỉ số phim, và lưu kết quả vào file edited_rating.csv

df = pd.read_csv('archive/rating.csv')

# chuyển user ids từ 1-based sang 0-based
df.userId = df.userId - 1

unique_movie_ids = set(df.movieId.values)
movie2idx = {}
count = 0
# Tạo một dictionary để ánh xạ movie ids sang movie indices
for movie_id in unique_movie_ids:
    movie2idx[movie_id] = count
    count += 1

df['movie_idx'] = df.apply(lambda row: movie2idx[row.movieId], axis=1)
df = df.drop(columns=['timestamp'])

df.to_csv('archive/edited_rating.csv', index=False)