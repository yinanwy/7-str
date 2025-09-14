import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pickle

# ==================== 1. 数据加载 ====================
raw_data = pd.read_csv(r"D:\桌面\旧浙江处理\浙江分析.csv")

# ==================== 2. 特征选择 ====================
selected_features = ['CW02', 'CR02', 'CP03', 'CR04', 'CS02', 'CP02', 'CS01']
X_raw = raw_data[selected_features]
y = raw_data.iloc[:, -1]  # 假设目标变量在最后一列

# ==================== 3. 划分训练集和测试集 ====================
# 先划分数据，避免数据泄露
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.25, random_state=42
)

# ==================== 4. 数据预处理 ====================
# 4.1 标准化处理（仅在训练集上fit，避免数据泄露）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)  # 测试集用训练集的scaler

# 4.2 L2正则化（同样只在训练集上fit）
normalizer = Normalizer(norm='l2')
X_train_processed = normalizer.fit_transform(X_train_scaled)
X_test_processed = normalizer.transform(X_test_scaled)  # 测试集用训练集的normalizer

# ==================== 5. 人工合成采样 (SMOTE) ====================
# **只在训练集上应用SMOTE，测试集保持不变**
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

# ==================== 6. 模型训练 ====================
# 使用随机森林
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)
model.fit(X_train_resampled, y_train_resampled)

# ==================== 7. 保存模型、预处理器和测试集 ====================
# 7.1 保存模型
with open('rf_model_4features.pkl', 'wb') as f:
    pickle.dump(model, f)

# 7.2 保存预处理器（重要！）
preprocessors = {
    'scaler': scaler,
    'normalizer': normalizer,
    'feature_names': selected_features  # 保存特征顺序
}
with open('preprocessors_4features.pkl', 'wb') as f:
    pickle.dump(preprocessors, f)

# 7.3 仅保存预处理后的测试集（X_test_processed 和 y_test）
test_data = {
    'X_test_processed': X_test_processed,  # 标准化+正则化后的测试集特征
    'y_test': y_test                       # 测试集标签
}
with open('processed_test_data.pkl', 'wb') as f:
    pickle.dump(test_data, f)

print("随机森林模型、预处理器和测试集保存完成！")
print(f"特征顺序：{selected_features}")