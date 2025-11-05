import os
import pickle
import logging
from transformers.trainer import TrainerState

# --- 配置日志 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 你需要修改的路径 ---
CHECKPOINT_DIR = "fine_tune_checkpoints/m2m100_1_2b_multilingual/checkpoint-11000" 

# 2. 你想要生成的 .pkl 文件的完整路径
OUTPUT_PKL_FILE = "fine_tune_checkpoints/m2m100_1_2b_multilingual/training_log_history.pkl"


# 构造 trainer_state.json 文件的完整路径
state_json_path = os.path.join(CHECKPOINT_DIR, "trainer_state.json")

if not os.path.exists(state_json_path):
    logger.error(f"错误：在指定目录 {CHECKPOINT_DIR} 中找不到 'trainer_state.json'。")
    logger.error("请确保 CHECKPOINT_DIR 指向一个有效的检查点文件夹。")
else:
    try:
        logger.info(f"正在从 {state_json_path} 加载 TrainerState...")
        
        # 1. 从 JSON 文件加载状态
        trainer_state = TrainerState.load_from_json(state_json_path)
        
        # 2. 提取 log_history
        #    trainer_state.log_history 就是你需要的那个列表
        log_history = trainer_state.log_history
        
        logger.info(f"成功提取 log_history (共 {len(log_history)} 条记录)")
        
        # 3. 使用 pickle 将 log_history 保存到 .pkl 文件
        with open(OUTPUT_PKL_FILE, "wb") as f:
            pickle.dump(log_history, f)
            
        logger.info(f"已成功将 Log History 重新保存到: {OUTPUT_PKL_FILE}")

    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")