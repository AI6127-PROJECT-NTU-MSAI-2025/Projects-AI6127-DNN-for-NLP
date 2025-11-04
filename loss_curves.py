import pickle
import pandas as pd
import matplotlib.pyplot as plt

##在这里输入check_point文件夹位置
checkpoint_path = "fine_tune_checkpoints/mt5_finetune"

with open(checkpoint_path+"/training_log_history.pkl", "rb") as f:
    log_history = pickle.load(f)

# 2. 转换为 DataFrame
df = pd.DataFrame(log_history)
#df.to_csv("log_history.csv", index=False)

print(df.head(10))
print(df.columns)

# 3. 提取训练 Loss (每一步)
train_df = df.dropna(subset=['loss']) # loss 列有值的就是训练步骤的日志
# 提取评估 Loss (每次评估)
eval_df = df.dropna(subset=['eval_loss']) # eval_loss 列有值的就是评估步骤的日志

# 4. 绘图
plt.figure(figsize=(12, 6))

# 绘制训练 Loss
plt.plot(train_df['step'], train_df['loss'], label='Training Loss (Step)', marker='.', linestyle='--', alpha=0.6)

# 绘制评估 Loss
if not eval_df.empty:
    plt.plot(eval_df['step'], eval_df['eval_loss'], label='Evaluation Loss (Checkpoint)', marker='o', linestyle='-', color='red')

plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss Curve')
plt.legend()
plt.grid(True)
plt.show()
