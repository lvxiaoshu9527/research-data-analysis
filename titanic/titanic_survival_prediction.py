# =============================================================================
#
#  泰坦尼克号生存预测挑战 - (终极竞赛版)
#  策略: 高级特征工程 + 模型堆叠(Stacking)
#  注意: 运行此代码前，请确保已安装所有需要的库:
#        pip install pandas scikit-learn lightgbm
#
# =============================================================================

# --- 步骤 0: 导入我们的“终极武器库” ---
import pandas as pd
import numpy as np
import re

# 模型
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as lgb

# 模型融合工具
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# 其他工具
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

print(">>> 终极武器库已加载，开始执行冲榜任务...")

# --- 步骤 1: 加载数据 ---
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    print(">>> 'train.csv' 和 'test.csv' 文件加载成功。")
except FileNotFoundError as e:
    print(f"错误: 找不到文件 {e.filename}。")
    exit()

# --- 步骤 2: 高级特征工程 ---
print(">>> 开始进行高级特征工程...")

# 合并数据集以统一处理
full_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
test_passenger_ids = full_df[full_df['Survived'].isnull()]['PassengerId']

# 2a. 称谓 (Title)
full_df['Title'] = full_df['Name'].apply(lambda x: re.search(' ([A-Za-z]+)\.', x).group(1))
title_mapping = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare", "Mlle": "Miss",
    "Countess": "Rare", "Ms": "Miss", "Lady": "Rare", "Jonkheer": "Rare",
    "Don": "Rare", "Dona": "Rare", "Mme": "Mrs", "Capt": "Rare", "Sir": "Rare"
}
full_df['Title'] = full_df['Title'].map(title_mapping)

# 2b. 家庭规模 (FamilySize)
full_df['FamilySize'] = full_df['SibSp'] + full_df['Parch'] + 1
full_df['IsAlone'] = (full_df['FamilySize'] == 1).astype(int)

# 2c. 甲板号 (Deck) - 核心高级特征
full_df['Deck'] = full_df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'U')

# 2d. 票价 (Fare) - 填充与分箱
full_df['Fare'] = full_df.groupby(['Pclass', 'Title'])['Fare'].transform(lambda x: x.fillna(x.median()))
full_df['FareBin'] = pd.qcut(full_df['Fare'], 4, labels=False)

# 2e. 年龄 (Age) - 填充与分箱
full_df['Age'] = full_df.groupby(['Title'])['Age'].transform(lambda x: x.fillna(x.median()))
full_df['AgeBin'] = pd.qcut(full_df['Age'], 5, labels=False)

# 2f. 登船港口 (Embarked)
full_df['Embarked'] = full_df['Embarked'].fillna(full_df['Embarked'].mode()[0])

# 2g. 创建组合特征
full_df['Sex_Pclass'] = full_df['Sex'] + "_" + full_df['Pclass'].astype(str)

# --- 步骤 3: 特征转换 ---
print(">>> 特征工程完毕，开始进行最终转换...")

# 删除不再需要的原始列
full_df.drop(['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1, inplace=True)

# 将分类变量转换为数值变量 (独热编码)
full_df = pd.get_dummies(full_df, columns=['Sex', 'Embarked', 'Title', 'Deck', 'Sex_Pclass'], drop_first=True)

# 分离数据
train_final = full_df[full_df['Survived'].notnull()]
test_final = full_df[full_df['Survived'].isnull()].drop('Survived', axis=1)

X = train_final.drop(['Survived', 'PassengerId'], axis=1)
y = train_final['Survived'].astype(int)
X_test = test_final.drop(['PassengerId'], axis=1)

# 特征缩放
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# --- 步骤 4: 构建与训练模型堆叠(Stacking) ---
print(">>> 开始构建与训练模型堆叠(Stacking)...")

# 定义第一层的基础模型（我们的英雄联盟）
estimators = [
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)),
    ('lgbm', lgb.LGBMClassifier(random_state=42)),
    ('svc', SVC(probability=True, random_state=42))
]

# 定义第二层的元模型（我们的神盾局局长）
# 它将学习如何组合第一层模型的预测
meta_model = LogisticRegression()

# 构建 Stacking 分类器
# cv=5 表示用5折交叉验证来生成第一层模型的预测，以防止数据泄露
stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_model,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
)

# 训练我们的终极模型！
stacking_model.fit(X, y)

print(">>> 模型堆叠训练完毕！")

# --- 步骤 5: 预测并生成最终提交文件 ---
print(">>> 开始对测试数据进行最终预测...")

# 进行预测
predictions = stacking_model.predict(X_test)

# 创建提交文件
submission_df = pd.DataFrame({
    'PassengerId': test_passenger_ids.astype(int),
    'Survived': predictions.astype(int)
})

submission_df.to_csv('titanic_submission_stacking.csv', index=False)

print(">>> 冲榜任务完成！最终预测结果已保存至 'titanic_submission_stacking.csv' 文件。")
print(">>> 这是我们能打造的最强阵容了，祝你在Kaggle上取得好成绩！")
