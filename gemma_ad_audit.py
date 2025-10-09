"""
广告审核治理系统 - Gemma大模型实现
使用本地部署的Ollama Gemma模型进行广告审核，并与LightGBM模型进行对比

依赖：
pip install scikit-learn matplotlib pandas numpy requests pillow ollama
"""

import os
import re
import json
import pickle
import joblib
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import requests
from sklearn.metrics import precision_recall_curve, average_precision_score, auc, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning)

# Ollama API配置
OLLAMA_API_URL = "http://localhost:11434/api"
GEMMA_MODEL = "gemma3n"

class GemmaAdAudit:
    """使用Ollama Gemma模型进行广告审核的类"""
    
    def __init__(self, model_name=GEMMA_MODEL):
        """初始化Gemma广告审核模型
        
        Args:
            model_name: Ollama模型名称
        """
        self.model_name = model_name
        self.threshold = 0.5  # 默认分类阈值
        self.best_threshold = None
        self.feature_semantics = {}
        
    def generate_prompt(self, ad_text, ad_image_path=None):
        """生成用于Gemma模型的提示
        
        Args:
            ad_text: 广告文本
            ad_image_path: 广告图片路径（可选）
            
        Returns:
            str: 格式化的提示
        """
        prompt = f"""你是一个专业的广告审核员。请严格分析以下广告内容，判断其违规的概率。
违规广告的特征包括但不限于：
1. 虚假宣传或夸大产品功效（如“100%中奖”、“一夜暴富”）。
2. 诱导性内容（如“点击免费领取”、“限时秒杀”）。
3. 宣传违禁品或非法服务（如赌博、色情、毒品、非法贷款）。
4. 涉及人身攻击、歧视或不当言论。
5. 侵犯他人隐私或知识产权。

以下是一些示例：
广告文本: "免费领取iPhone 15 Pro Max，数量有限，先到先得！"
违规概率: 0.95

广告文本: "高品质T恤，纯棉材质，舒适透气，夏季必备。"
违规概率: 0.05

现在，请你判断以下广告内容：

广告文本: {ad_text}

请仅输出一个介于0到1之间的浮点数作为违规概率，例如 "违规概率: 0.78"，不要包含其他解释。"""
        
        return prompt
    
    def query_ollama(self, prompt):
        """向Ollama API发送请求
        
        Args:
            prompt: 提示文本
            
        Returns:
            str: 模型响应
        """
        try:
            response = requests.post(
                f"{OLLAMA_API_URL}/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            print(f"Ollama API请求失败: {e}")
            return ""
    
    def predict_proba(self, texts, image_paths=None):
        """预测广告违规概率
        
        Args:
            texts: 广告文本列表
            image_paths: 广告图片路径列表（可选）
            
        Returns:
            np.array: 违规概率数组 shape=(n_samples, 2)
        """
        n_samples = len(texts)
        probas = np.zeros((n_samples, 2))
        
        for i, text in enumerate(texts):
            image_path = None if image_paths is None else image_paths[i]
            prompt = self.generate_prompt(text, image_path)
            response = self.query_ollama(prompt)
            
            # 解析响应，提取概率
            try:
                # 查找 "违规概率: " 后面的浮点数
                match = re.search(r"违规概率:\s*([0-1]?\.\d+)", response)
                if match:
                    gemma_proba_value = float(match.group(1))
                else:
                    # 如果没有匹配到，尝试查找 "违规" 或 "合规"，并赋予默认概率
                    if "违规" in response:
                        gemma_proba_value = 0.9  # 默认高概率违规
                    elif "合规" in response:
                        gemma_proba_value = 0.1  # 默认低概率违规
                    else:
                        gemma_proba_value = 0.5  # 无法判断时，默认中性概率
            except ValueError:
                gemma_proba_value = 0.5  # 解析失败时，默认中性概率

            probas[i, 1] = gemma_proba_value
            probas[i, 0] = 1 - gemma_proba_value
                
            # 打印进度
            if (i + 1) % 10 == 0 or i == n_samples - 1:
                print(f"已处理 {i+1}/{n_samples} 条广告")
                
        return probas
    
    def predict(self, texts, image_paths=None):
        """预测广告是否违规
        
        Args:
            texts: 广告文本列表
            image_paths: 广告图片路径列表（可选）
            
        Returns:
            np.array: 预测标签 (0=合规, 1=违规)
        """
        probas = self.predict_proba(texts, image_paths)
        threshold = self.best_threshold if self.best_threshold is not None else self.threshold
        return (probas[:, 1] >= threshold).astype(int)
    
    def optimize_threshold(self, y_true, y_proba, cost_ratio=5):
        """优化分类阈值
        
        Args:
            y_true: 真实标签
            y_proba: 预测概率
            cost_ratio: 漏报成本/误报成本比率
            
        Returns:
            float: 最优阈值
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        
        # 计算每个阈值的成本
        costs = []
        for i in range(len(precision)):
            if precision[i] == 0:  # 避免除零错误
                cost = float('inf')
            else:
                # 计算误报和漏报
                fp = (1 - precision[i]) * recall[i] * sum(y_true) / precision[i]
                fn = (1 - recall[i]) * sum(y_true)
                # 计算总成本
                cost = fp + cost_ratio * fn
            costs.append(cost)
        
        # 找到最小成本对应的阈值
        best_idx = np.argmin(costs)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        
        self.best_threshold = best_threshold
        return best_threshold
    
    def save_model(self, filepath):
        """保存模型配置
        
        Args:
            filepath: 保存路径
        """
        model_data = {
            "model_name": self.model_name,
            "threshold": self.threshold,
            "best_threshold": self.best_threshold,
            "feature_semantics": self.feature_semantics
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"模型配置已保存至 {filepath}")
    
    def load_model(self, filepath):
        """加载模型配置
        
        Args:
            filepath: 模型文件路径
        """
        try:
            import joblib
            model_data = joblib.load(filepath)
            
            self.model_name = model_data.get("model_name", self.model_name)
            self.threshold = model_data.get("threshold", self.threshold)
            self.best_threshold = model_data.get("best_threshold", self.best_threshold)
            self.feature_semantics = model_data.get("feature_semantics", {})
            
            print(f"模型配置已从 {filepath} 加载")
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False


def _extract_lgbm_features(text, svd_input_dim):
    """
    从文本中提取LightGBM模型所需的特征。
    模拟prediction_with_model.py中的特征构建逻辑。
    """
    features = np.zeros(svd_input_dim)
    
    if text:
        # 设置文本长度特征（放在倒数第三个位置）
        features[-3] = len(text)
        # 设置一些关键词特征
        if "免费" in text or "free" in text.lower():
            features[-5] = 1
        if "限时" in text or "limited" in text.lower():
            features[-6] = 1
    
    # 假设没有图片，所以图片特征标志为0
    features[-2] = 0
    
    # 设置一个默认的数值特征（放在最后一个位置）
    features[-1] = 0.05  # 默认CTR值
    
    return features


def load_lightgbm_model(model_path):
    """加载LightGBM模型
    
    Args:
        model_path: 模型文件路径
        
    Returns:
        dict: 模型数据
    """
    try:
        with open(model_path, 'rb') as f:
            model_data = joblib.load(f)
        return model_data
    except Exception as e:
        print(f"加载LightGBM模型失败: {e}")
        return None


def compare_models(df, lgbm_model_path, gemma_model_path=None):
    """比较LightGBM和Gemma模型性能
    
    Args:
        df: 数据集
        lgbm_model_path: LightGBM模型路径
        gemma_model_path: Gemma模型路径（可选）
    """
    # 加载LightGBM模型
    lgbm_data = load_lightgbm_model(lgbm_model_path)
    if lgbm_data is None:
        print("无法加载LightGBM模型，比较终止")
        return
    
    # 准备数据
    X_text = df['ad_text'].values
    y_true = df['label'].values
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y_true, test_size=0.2, random_state=42, stratify=y_true
    )
    
    # 使用LightGBM模型预测
    print("使用LightGBM模型进行预测...")
    # 这里需要根据实际情况处理特征提取
    clf = lgbm_data['clf']
    svd = lgbm_data['svd']
    lgbm_threshold = lgbm_data.get('threshold', 0.39)

    # 获取SVD的输入维度
    svd_input_dim = svd.n_features_in_

    # 为LightGBM模型提取特征并进行预测
    lgbm_features = []
    for text in X_test:
        lgbm_features.append(_extract_lgbm_features(text, svd_input_dim))
    lgbm_features = np.array(lgbm_features)

    # 转换特征并预测
    lgbm_features_transformed = svd.transform(lgbm_features)
    lgbm_proba = clf.predict_proba(lgbm_features_transformed)[:, 1]
    
    # 初始化Gemma模型
    gemma_model = GemmaAdAudit()
    if gemma_model_path and os.path.exists(gemma_model_path):
        gemma_model.load_model(gemma_model_path)
    
    # 使用Gemma模型预测
    print("使用Gemma模型进行预测...")
    
    X_test_gemma = X_test
    y_test_gemma = y_test
    
    gemma_probas = gemma_model.predict_proba(X_test_gemma)
    gemma_proba = gemma_probas[:, 1]
    
    # 计算PR曲线
    lgbm_precision, lgbm_recall, lgbm_thresholds = precision_recall_curve(y_test, lgbm_proba)
    gemma_precision, gemma_recall, gemma_thresholds = precision_recall_curve(y_test_gemma, gemma_proba)
    
    lgbm_ap = average_precision_score(y_test, lgbm_proba)
    gemma_ap = average_precision_score(y_test_gemma, gemma_proba)
    
    # 绘制PR曲线对比
    plt.figure(figsize=(10, 8))
    plt.plot(lgbm_recall, lgbm_precision, label=f'LightGBM (AP={lgbm_ap:.3f})')
    plt.plot(gemma_recall, gemma_precision, label=f'Gemma (AP={gemma_ap:.3f})')
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('LightGBM vs Gemma PR曲线对比')
    plt.legend()
    plt.grid(True)
    
    # 保存图像
    os.makedirs('models', exist_ok=True)
    plt.savefig(os.path.join('models', 'pr_comparison_gemma.png'))
    print(f"PR曲线对比已保存至 models/pr_comparison_gemma.png")
    
    # 计算ROC曲线
    lgbm_fpr, lgbm_tpr, _ = roc_curve(y_test, lgbm_proba)
    gemma_fpr, gemma_tpr, _ = roc_curve(y_test_gemma, gemma_proba)
    
    lgbm_auc = auc(lgbm_fpr, lgbm_tpr)
    gemma_auc = auc(gemma_fpr, gemma_tpr)
    
    # 绘制ROC曲线对比
    plt.figure(figsize=(10, 8))
    plt.plot(lgbm_fpr, lgbm_tpr, label=f'LightGBM (AUC={lgbm_auc:.3f})')
    plt.plot(gemma_fpr, gemma_tpr, label=f'Gemma (AUC={gemma_auc:.3f})')
    plt.xlabel('假正例率')
    plt.ylabel('真正例率')
    plt.title('LightGBM vs Gemma ROC曲线对比')
    plt.legend()
    plt.grid(True)
    plt.plot([0, 1], [0, 1], 'k--')
    
    # 保存图像
    plt.savefig(os.path.join('models', 'roc_comparison_gemma.png'))
    print(f"ROC曲线对比已保存至 models/roc_comparison_gemma.png")
    
    # 优化Gemma模型阈值
    best_threshold = gemma_model.optimize_threshold(y_test_gemma, gemma_proba)
    print(f"Gemma模型最优阈值: {best_threshold:.4f}")
    
    # 保存Gemma模型
    gemma_model.save_model(os.path.join('models', 'gemma_ad_audit_model.pkl'))


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='广告审核 - Gemma模型与LightGBM对比')
    parser.add_argument('--data', type=str, default='data.csv', help='数据集路径')
    parser.add_argument('--lgbm_model', type=str, default='models/ad_audit_model.pkl', help='LightGBM模型路径')
    parser.add_argument('--gemma_model', type=str, default=None, help='Gemma模型路径（可选）')
    parser.add_argument('--predict', action='store_true', help='使用模型进行预测')
    parser.add_argument('--compare', action='store_true', help='比较两个模型性能')
    parser.add_argument('--text', type=str, default=None, help='要预测的广告文本')
    
    args = parser.parse_args()
    
    # 加载数据
    try:
        print(f"正在从{args.data}加载数据...")
        df = pd.read_csv(args.data)
        print(f"成功加载数据，共{len(df)}条记录")
    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    
    if args.compare:
        # 比较模型性能
        compare_models(df, args.lgbm_model, args.gemma_model)
    elif args.predict and args.text:
        # 使用Gemma模型进行预测
        gemma_model = GemmaAdAudit()
        if args.gemma_model and os.path.exists(args.gemma_model):
            gemma_model.load_model(args.gemma_model)
        
        result = gemma_model.predict([args.text])[0]
        print(f"广告文本: {args.text}")
        print(f"预测结果: {'违规' if result == 1 else '合规'}")
    else:
        print("请指定操作: --compare 或 --predict --text '广告文本'")


if __name__ == "__main__":
    main()