data_dir = "CelebA_Spoof/Data/test"
import os
import shutil
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, classification_report, confusion_matrix, auc, roc_curve
import matplotlib.pyplot as plt


def bench_mark(y_test, y_pred, infer_time):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")

def roc_curve_plots(y_true, y_pred, title='ROC Curve', figsize=(12, 9)):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.2f})", color="blue")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")  # Random baseline
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

def get_image_paths_and_labels(folder, label):
    return [(os.path.join(folder, file), label) for file in os.listdir(folder) if file.endswith(".png")]

def data_gen(data_dir):
    live_folder = os.path.join(data_dir, "live")
    spoof_folder = os.path.join(data_dir, "spoof")

    # Lấy danh sách ảnh và gắn nhãn
    def get_image_paths_and_labels(folder, label):
        return [(os.path.join(folder, file), label) for file in os.listdir(folder) if file.endswith(".png")]

    # Tạo danh sách ảnh và nhãn
    live_data = get_image_paths_and_labels(live_folder, 1)  # Label 1 cho live
    spoof_data = get_image_paths_and_labels(spoof_folder, 0)  # Label 0 cho spoof

    # Gộp dữ liệu và tạo DataFrame
    data = pd.DataFrame(live_data + spoof_data, columns=["image_path", "label"])
    data.to_csv('C:/Users/Admin/Documents/VSCode/Code_Anti_Face_Spoofing/data.csv')
    print('CSV CREATED')
    return data

# Đường dẫn tới thư mục gốc
data_folder = "CelebA_Spoof/Data/test"
output_live = "CelebA_Spoof/Data/live"
output_spoof = "CelebA_Spoof/Data/spoof"

# Tạo các thư mục đích nếu chưa tồn tại
os.makedirs(output_live, exist_ok=True)
os.makedirs(output_spoof, exist_ok=True)

# Duyệt qua các thư mục con trong `data`
for id_folder in os.listdir(data_folder):
    id_path = os.path.join(data_folder, id_folder)
    
    # Kiểm tra xem có phải thư mục không
    if os.path.isdir(id_path):
        # Đường dẫn tới các thư mục live và spoof
        live_path = os.path.join(id_path, "live")
        spoof_path = os.path.join(id_path, "spoof")
        
        # Copy ảnh từ live folder
        if os.path.exists(live_path):
            for file_name in os.listdir(live_path):
                if file_name.endswith(".png"):  # Chỉ lấy ảnh .png
                    src = os.path.join(live_path, file_name)
                    dst = os.path.join(output_live, f"{id_folder}_{file_name}")  # Thêm tiền tố ID để tránh trùng tên
                    shutil.copy(src, dst)
        
        # Copy ảnh từ spoof folder
        if os.path.exists(spoof_path):
            for file_name in os.listdir(spoof_path):
                if file_name.endswith(".png"):  # Chỉ lấy ảnh .png
                    src = os.path.join(spoof_path, file_name)
                    dst = os.path.join(output_spoof, f"{id_folder}_{file_name}")  # Thêm tiền tố ID để tránh trùng tên
                    shutil.copy(src, dst)

print("Hoàn thành di chuyển ảnh!")
