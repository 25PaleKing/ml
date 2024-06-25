import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import multiprocessing

# 加载训练数据
train_data = pd.read_csv('weibo_train_data.txt', sep='\t', header=None,
                         names=['uid', 'mid', 'time', 'forward_count', 'comment_count', 'like_count', 'content'])

# 加载预测数据
predict_data = pd.read_csv('weibo_predict_data.txt', sep='\t', header=None, names=['uid', 'mid', 'time', 'content'])

# 数据预处理函数
def preprocess_data(data, is_train=True):
    data['time'] = pd.to_datetime(data['time'])
    data['day'] = data['time'].dt.day
    data['month'] = data['time'].dt.month
    data['year'] = data['time'].dt.year
    data['hour'] = data['time'].dt.hour

    # 处理缺失值
    data['content'] = data['content'].fillna('')  # 将缺失值填充为空字符串
    data['content_length'] = data['content'].apply(lambda x: len(x))

    if is_train:
        # 对于训练数据，填充目标变量的缺失值
        data['forward_count'] = data['forward_count'].fillna(0)
        data['comment_count'] = data['comment_count'].fillna(0)
        data['like_count'] = data['like_count'].fillna(0)

    return data

# 并行处理数据预处理过程
num_cores = multiprocessing.cpu_count()  # 获取计算机的核心数
train_data_processed = Parallel(n_jobs=num_cores)(delayed(preprocess_data)(data) for data in [train_data])

# 合并处理后的数据
train_data_processed = pd.concat(train_data_processed)

# 处理预测数据
predict_data_processed = preprocess_data(predict_data, is_train=False)

# 选择特征和目标变量
X = train_data_processed[['day', 'month', 'year', 'hour', 'content_length']]
y = train_data_processed[['forward_count', 'comment_count', 'like_count']]

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
predict_data_scaled = scaler.transform(predict_data_processed[['day', 'month', 'year', 'hour', 'content_length']])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 检查训练集和测试集中是否存在 NaN 值
if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)) or np.any(np.isnan(X_test)) or np.any(np.isnan(y_test)):
    print("NaN values found in the train/test data.")
else:
    print("No NaN values in the train/test data.")

# 定义随机森林模型
rf_forward = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_comment = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_like = RandomForestRegressor(random_state=42, n_jobs=-1)

# 定义梯度提升回归模型
gb_forward = GradientBoostingRegressor(random_state=42)
gb_comment = GradientBoostingRegressor(random_state=42)
gb_like = GradientBoostingRegressor(random_state=42)

# 并行训练模型
print("Training models...")
models = {
    'Forward_RF': rf_forward,
    'Comment_RF': rf_comment,
    'Like_RF': rf_like,
    'Forward_GB': gb_forward,
    'Comment_GB': gb_comment,
    'Like_GB': gb_like
}

for name, model in models.items():
    if 'RF' in name:
        model.fit(X_train, y_train['forward_count' if 'Forward' in name else 'comment_count' if 'Comment' in name else 'like_count'])
    else:
        model.fit(X_train, y_train['forward_count' if 'Forward' in name else 'comment_count' if 'Comment' in name else 'like_count'])

# 预测
y_preds = {
    'Forward_RF': models['Forward_RF'].predict(X_test),
    'Comment_RF': models['Comment_RF'].predict(X_test),
    'Like_RF': models['Like_RF'].predict(X_test),
    'Forward_GB': models['Forward_GB'].predict(X_test),
    'Comment_GB': models['Comment_GB'].predict(X_test),
    'Like_GB': models['Like_GB'].predict(X_test)
}

# 计算误差
maes = {
    name: mean_absolute_error(y_test['forward_count' if 'Forward' in name else 'comment_count' if 'Comment' in name else 'like_count'], y_pred)
    for name, y_pred in y_preds.items()
}

# 打印误差
for name, mae in maes.items():
    print(f'MAE for {name}: {mae}')

# 预测新数据
print("Predicting for new data...")
predictions = {
    'Forward_RF': models['Forward_RF'].predict(predict_data_scaled),
    'Comment_RF': models['Comment_RF'].predict(predict_data_scaled),
    'Like_RF': models['Like_RF'].predict(predict_data_scaled),
    'Forward_GB': models['Forward_GB'].predict(predict_data_scaled),
    'Comment_GB': models['Comment_GB'].predict(predict_data_scaled),
    'Like_GB': models['Like_GB'].predict(predict_data_scaled)
}

# 构建结果数据框
results = pd.DataFrame({
    'uid': predict_data['uid'],
    'mid': predict_data['mid'],
    'forward_count': predictions['Forward_RF'].astype(int) + predictions['Forward_GB'].astype(int),
    'comment_count': predictions['Comment_RF'].astype(int) + predictions['Comment_GB'].astype(int),
    'like_count': predictions['Like_RF'].astype(int) + predictions['Like_GB'].astype(int)
})

# 保存结果
results.to_csv('weibo_result_data.txt', sep='\t', header=False, index=False)