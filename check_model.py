import os
import pickle
import sys
import joblib

model_path = os.path.join('models', 'ad_audit_model.pkl')

print(f"正在检查模型文件: {model_path}")

if not os.path.exists(model_path):
    print("错误: 模型文件不存在！请确保 ad_audit_model.pkl 在 models 文件夹中。")
    sys.exit(1)

try:
    model_data = joblib.load(model_path)
    print("模型文件加载成功！")
    # 可以进一步检查 model_data 的类型或内容
    print(f"加载的模型数据类型: {type(model_data)}")
except pickle.UnpicklingError as e:
    print(f"错误: 模型文件损坏或格式不正确。Pickle 解包失败: {e}")
    print("建议: 尝试重新运行 ad_audit_pipeline.py 来重新生成模型文件。")
except Exception as e:
    print(f"加载模型文件时发生未知错误: {e}")
    print("建议: 尝试重新运行 ad_audit_pipeline.py 来重新生成模型文件。")