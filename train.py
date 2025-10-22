import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ==========================
# 1️⃣ LOAD DỮ LIỆU
# ==========================
data = np.load("features_aug.npz")
X = data["X"]
y = data["y"]

print("✅ Dữ liệu:", X.shape, y.shape)

# ==========================
# 2️⃣ CHUẨN HÓA
# ==========================
X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-6
X_scaled = (X - X_mean) / X_std

# ==========================
# 3️⃣ CHIA TẬP
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

classes = np.unique(y)
num_classes = len(classes)
num_features = X.shape[1]

print("🔹 Số lớp:", num_classes)
print("🔹 Số đặc trưng:", num_features)


# ==========================
# 4️⃣ HÀM HỖ TRỢ
# ==========================
def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def one_hot(y, num_classes):
    y_oh = np.zeros((len(y), num_classes))
    for i, val in enumerate(y):
        y_oh[i, int(val)] = 1
    return y_oh


# ==========================
# 5️⃣ HUẤN LUYỆN SOFTMAX
# ==========================
W = np.zeros((num_features, num_classes))
b = np.zeros((1, num_classes))

lr = 0.01
epochs = 500
y_train_oh = one_hot(y_train, num_classes)

for epoch in range(epochs):
    # Forward
    Z = np.dot(X_train, W) + b
    A = softmax(Z)

    # Loss (Cross-Entropy)
    loss = -np.mean(np.sum(y_train_oh * np.log(A + 1e-9), axis=1))

    # Gradient
    dW = np.dot(X_train.T, (A - y_train_oh)) / len(X_train)
    db = np.mean(A - y_train_oh, axis=0, keepdims=True)

    # Update
    W -= lr * dW
    b -= lr * db

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")

# ==========================
# 6️⃣ ĐÁNH GIÁ
# ==========================
Z_test = np.dot(X_test, W) + b
A_test = softmax(Z_test)
y_pred = np.argmax(A_test, axis=1)

print("\n📊 Kết quả test:")
print(classification_report(y_test, y_pred))
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Độ chính xác: {acc * 100:.2f}%")

# ==========================
# 7️⃣ LƯU MODEL & SCALER
# ==========================
joblib.dump({"W": W, "b": b, "classes": classes}, "softmax_model_best.pkl")
joblib.dump({"X_mean": X_mean, "X_std": X_std}, "scale.pkl")

print("💾 Đã lưu model vào 'softmax_model_best.pkl'")
print("💾 Đã lưu scaler vào 'scale.pkl'")
