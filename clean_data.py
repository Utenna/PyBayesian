# 导入数据加载和处理包：pandas
import pandas as pd
# 导入数字和向量处理包：numpy
import numpy as np
# 导入基本绘图工具：matplotlib
import matplotlib.pyplot as plt

# 使用 pandas 导入示例数据
dx = pd.read_csv("/Users/ss/Documents/bayes_2024/2024/data/replicated_language.csv")

# 创建一个字典，键为 study2_id，值为对应的 posemo 和 certain
posemo_dict = dx.set_index('study2_id')['posemo'].to_dict()
certain_dict = dx.set_index('study2_id')['certain'].to_dict()

# 使用 stu_0 列的值在 study2_id 中找到对应的 posemo 和 certain，并填充 posemo_re 和 certain_re 列
dx['posemo_re'] = dx['stu_0'].map(posemo_dict)
dx['certain_re'] = dx['stu_0'].map(certain_dict)

# 删除包含 NA 值的行
dx_cleaned = dx.dropna()

# 输出结果
print(dx_cleaned)

# 将预处理后的结果保存到 replicated_language.csv 文件中
dx_cleaned.to_csv("/Users/ss/Documents/bayes_2024/2024/data/replicated_language_cleaned.csv", index=False)

# 输出提示信息
print("初步预处理后的数据已成功保存到 replicated_language_cleaned.csv")

# 删除 'study2_id', 'posemo', 'certain' 列
dx_cleaned = dx_cleaned.drop(columns=['study2_id', 'posemo', 'certain'])

# 保存处理后的数据到 CSV 文件
dx_cleaned.to_csv("/Users/ss/Documents/bayes_2024/2024/data/replicated_language_cleaned.csv", index=False)

# 输出提示信息
print("已删除指定列，并将处理后的数据保存到 replicated_language_cleaned.csv")

# 重命名列 stu_0 -> study_id, posemo_re -> posemo, certain_re -> certain
dx_cleaned = dx_cleaned.rename(columns={'stu_0': 'study_id', 'posemo_re': 'posemo', 'certain_re': 'certain'})

# 保存处理后的数据到 CSV 文件
dx_cleaned.to_csv("/Users/ss/Documents/bayes_2024/2024/data/replicated_language_cleaned.csv", index=False)

# 输出提示信息
print("已重命名列，并将处理后的数据保存到 replicated_language_cleaned.csv")