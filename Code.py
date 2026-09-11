# =====================================================================
# BƯỚC 1: KHỞI TẠO MÔI TRƯỜNG VÀ ĐỌC DỮ LIỆU
# =====================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Đọc bộ dữ liệu
# Lưu ý: Đảm bảo file csv nằm cùng thư mục với file code
df = pd.read_csv('Metro_Interstate_Traffic_Volume.csv')

# Xem trước 5 dòng đầu tiên
print("Dữ liệu ban đầu:")
print(df.head())

# =====================================================================
# BƯỚC 2: TIỀN XỬ LÝ DỮ LIỆU & KHAI PHÁ ĐẶC TRƯNG (FEATURE ENGINEERING)
# =====================================================================
print("\n--- Bắt đầu Tiền xử lý dữ liệu ---")

# TÓM LƯỢC DỮ LIỆU
print("\nBảng tóm lược dữ liệu (describe):")
print(df.describe())

# 1. Ép kiểu cột date_time sang định dạng datetime chuẩn của Pandas
df['date_time'] = pd.to_datetime(df['date_time'])

# 2. Bóc tách các đặc trưng thời gian có tính chu kỳ
df['Hour'] = df['date_time'].dt.hour
df['DayOfWeek'] = df['date_time'].dt.dayofweek  # 0 là Thứ 2, 6 là Chủ nhật
df['Month'] = df['date_time'].dt.month

# 3. Tạo biến nhị phân phân biệt ngày đi làm và ngày nghỉ cuối tuần
df['Is_Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# 5. Làm sạch dữ liệu: Lọc bỏ lỗi nhiệt độ 0K và lượng mưa cực đoan (9831.3 mm)
initial_rows = len(df)
df = df[(df['temp'] > 0) & (df['rain_1h'] < 9800)]
print(f"Đã xóa {initial_rows - len(df)} dòng dữ liệu lỗi (nhiệt độ 0K và mưa bất thường).")

# =====================================================================
# BƯỚC 3: PHÂN TÍCH MÔ TẢ (EDA) & TRỰC QUAN HÓA
# =====================================================================
# Biểu đồ 1: Lưu lượng giao thông trung bình theo giờ (Phân biệt cuối tuần)
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='Hour', y='traffic_volume', hue='Is_Weekend', palette='Set1', marker='o')
plt.title('Lưu lượng giao thông trung bình theo giờ')
plt.xlabel('Giờ trong ngày')
plt.ylabel('Lưu lượng xe (Volume)')
plt.xticks(range(0, 24))
plt.grid(True)
plt.show()

# Biểu đồ 2: Ma trận tương quan (Heatmap)
plt.figure(figsize=(8, 6))
# Chỉ chọn các biến số để xem tương quan
numeric_cols = ['traffic_volume', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'Hour', 'DayOfWeek', 'Is_Weekend']
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Ma trận tương quan giữa các biến số')
plt.show()

# =====================================================================
# BƯỚC 4: MÃ HÓA DỮ LIỆU PHÂN LOẠI (ONE-HOT ENCODING)
# =====================================================================
# Cột weather_description quá chi tiết và có thể gây nhiễu, ta chỉ dùng weather_main và holiday
df_encoded = pd.get_dummies(df, columns=['weather_main', 'holiday'], drop_first=True)

# Lọc bỏ các cột không dùng làm input cho mô hình toán học
X = df_encoded.drop(columns=['traffic_volume', 'date_time', 'weather_description'])
y = df_encoded['traffic_volume']

# =====================================================================
# BƯỚC 5: CHIA TẬP TRAIN / TEST VÀ HUẤN LUYỆN MÔ HÌNH
# =====================================================================
# Chia tập dữ liệu 80% Huấn luyện, 20% Kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Bắt đầu Huấn luyện Mô hình ---")

# 1. Mô hình Baseline: Linear Regression (Hồi quy tuyến tính)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# 2. Mô hình Đề tài: Random Forest Regression
# n_estimators=100 nghĩa là xây dựng 100 cây quyết định
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Hoàn thành huấn luyện!")

# =====================================================================
# BƯỚC 6: ĐÁNH GIÁ VÀ SO SÁNH HIỆU SUẤT MÔ HÌNH
# =====================================================================
def evaluate_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"[{name}] MAE: {mae:.2f} | RMSE: {rmse:.2f} | R² Score: {r2:.4f}")

print("\n=== BẢNG KẾT QUẢ ĐÁNH GIÁ TẬP TEST ===")
evaluate_model("Linear Regression", y_test, y_pred_lr)
evaluate_model("Random Forest    ", y_test, y_pred_rf)

# =====================================================================
# BƯỚC 7: TRỰC QUAN HÓA MỨC ĐỘ QUAN TRỌNG CỦA ĐẶC TRƯNG (FEATURE IMPORTANCE)
# =====================================================================
# Lấy trọng số từ mô hình Random Forest
importances = rf_model.feature_importances_
feature_names = X.columns

# Đưa vào DataFrame và sắp xếp giảm dần
feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_df = feature_df.sort_values(by='Importance', ascending=False).head(10) # Lấy Top 10

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_df, x='Importance', y='Feature', palette='viridis')
plt.title('Top 10 đặc trưng quan trọng nhất (Random Forest)')
plt.xlabel('Mức độ quan trọng (Feature Importance)')
plt.ylabel('Đặc trưng')
plt.show()