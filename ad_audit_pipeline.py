"""
End-to-End 广告审核治理系统 Demo
包含：
1. 规则层 + OCR（真实实现）
2. 文本/图像特征建模 + SMOTE/class_weight
3. PR 曲线阈值优化
4. SHAP 模型解释
5. 端到端违规检测与解释

依赖：
pip install scikit-learn imbalanced-learn lightgbm shap sentence-transformers rapidfuzz pillow pytesseract matplotlib opencv-python requests
"""

import re, unicodedata, os, io, requests
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from PIL import Image, ImageFilter, ImageOps
import pytesseract

# ========== 1. 数据加载 ==========
# 从CSV文件加载数据
try:
    print("正在从data.csv加载数据...")
    df = pd.read_csv('data.csv')
    print(f"成功加载数据，共{len(df)}条记录")
    print(f"违规广告数量: {df['label'].sum()}, 正常广告数量: {len(df) - df['label'].sum()}")
except Exception as e:
    print(f"加载数据失败: {e}")
    print("将使用模拟数据代替...")
    # 如果加载失败，使用模拟数据作为备选
    np.random.seed(42)
    N = 2000
    texts = np.random.choice([
        "Free iPhone click now", "Limited offer discount shoes", "Educational course in AI",
        "Normal clothing ad", "High quality electronics sale", "Get rich quick!!!"
    ], size=N)
    ocr_variants = []
    for t in texts:
        if "Free" in t or "rich" in t:
            ocr_variants.append(t if np.random.rand() < 0.5 else "")
        else:
            ocr_variants.append("")
    ocr = np.array(ocr_variants)
    ctr = np.random.beta(1.2, 10, size=N)
    impr_1h = np.random.poisson(50, size=N)
    adv_hist_violation = np.random.binomial(1, 0.05, size=N)
    labels = np.array([1 if ("Free" in t or "rich" in t) else 0 for t in texts])
    flip_idx = np.random.choice(N, size=int(0.02*N), replace=False)
    labels[flip_idx] = 1 - labels[flip_idx]
    df = pd.DataFrame({
        "ad_text": texts,
        "ocr_text": ocr,
        "ctr": ctr,
        "impr_1h": impr_1h,
        "adv_hist_violation": adv_hist_violation,
        "label": labels
    })

# 数据预处理
print("正在进行数据预处理...")
# 确保所有必要的列都存在
required_columns = ['ad_text', 'ctr', 'impr_1h', 'adv_hist_violation', 'label']
for col in required_columns:
    if col not in df.columns:
        if col == 'ad_text':
            df[col] = df.get('ad_text', '')
        elif col in ['ctr', 'impr_1h', 'adv_hist_violation']:
            df[col] = df.get(col, 0)
        elif col == 'label':
            df[col] = df.get(col, 0)

# 确保OCR文本列存在
if 'ocr_text' not in df.columns:
    df['ocr_text'] = ''

# 填充缺失值
df['ad_text'] = df['ad_text'].fillna('')
df['ocr_text'] = df['ocr_text'].fillna('')
df['ctr'] = df['ctr'].fillna(0)
df['impr_1h'] = df['impr_1h'].fillna(0)
df['adv_hist_violation'] = df['adv_hist_violation'].fillna(0)

print("数据预处理完成")

# ========== 2. 图片处理与OCR功能 ==========
def download_image(url, timeout=5):
    """下载图片并转换为PIL图像对象"""
    try:
        resp = requests.get(url, timeout=timeout)
        return Image.open(io.BytesIO(resp.content)).convert('RGB')
    except Exception as e:
        print(f"图片下载失败: {e}")
        return None

def preprocess_for_ocr(img_pil):
    """OCR前的图像预处理"""
    if img_pil is None:
        return None
    try:
        img = img_pil.convert('L')  # 转灰度
        img = img.resize((int(img.width*1.5), int(img.height*1.5)), Image.BILINEAR)  # 放大
        img = ImageOps.autocontrast(img)  # 自动对比度
        img = img.filter(ImageFilter.MedianFilter(size=3))  # 中值滤波去噪
        return img
    except Exception as e:
        print(f"图像预处理失败: {e}")
        return img_pil

def ocr_image(img_pil):
    """OCR提取图像中的文本"""
    if img_pil is None:
        return ""
    try:
        data = pytesseract.image_to_data(img_pil, output_type=pytesseract.Output.DATAFRAME)
        text_pieces = []
        for _, row in data.dropna(subset=['text']).iterrows():
            txt = str(row['text']).strip()
            if txt:
                text_pieces.append(txt)
        full_text = " ".join(text_pieces)
        return full_text
    except Exception as e:
        print(f"OCR处理失败: {e}")
        return ""

# ========== 3. 规则层过滤 ==========
def normalize_text(s):
    if pd.isna(s) or s is None: 
        return ""
    s = unicodedata.normalize('NFKC', str(s))
    s = s.replace('\u200b', '')  # 零宽空格
    s = s.strip().lower()
    # 基本字符映射
    s = s.replace('＠','@').replace('＃','#')
    return s

def rule_check_text(text):
    text = normalize_text(text)
    if not text:
        return False, None
        
    # 精确/正则匹配
    rules = [r"100%中奖", r"free", r"get rich", r"免费", r"贷款.*无利息", r"zero\s*fee"]
    for p in rules:
        if re.search(p, text, flags=re.I):
            return True, f"regex:{p}"
            
    # 模糊检测（处理插空格/同音/字符替换）
    keywords = ["free", "iPhone", "中奖", "返现"]
    for kw in keywords:
        score = fuzz.partial_ratio(kw, text)
        if score > 85:  # 阈值可调
            return True, f"fuzzy:{kw}:{score}"
    return False, None

# 对模拟数据应用规则
df['rule_block'] = df['ad_text'].apply(lambda x: rule_check_text(x)[0]) | df['ocr_text'].apply(lambda x: rule_check_text(x)[0])
print("Rule-blocked samples:", df['rule_block'].sum())

# ========== 4. 特征提取（文本嵌入 + 图像特征 + 数值特征） ==========
from sentence_transformers import SentenceTransformer
import os

# 设置更长的超时时间和离线模式支持
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'  # 禁用符号链接警告
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # 启用离线模式，避免下载模型

# 文本特征提取
try:
    print("正在加载文本模型...")
    # 使用本地缓存模型，避免每次下载
    text_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    docs = (df['ad_text'].fillna('') + " " + df['ocr_text'].fillna('')).tolist()
    text_emb = text_model.encode(docs, show_progress_bar=True, batch_size=32)
    print(f"文本特征提取完成，维度: {text_emb.shape}")
except Exception as e:
    print(f"文本模型加载失败: {e}")
    print("使用随机特征向量替代...")
    # 使用随机向量替代，确保代码可以继续运行
    text_emb = np.random.randn(len(df), 384) * 0.01  # MiniLM模型输出维度为384

# 图像特征提取
try:
    print("正在加载图像模型...")
    img_model = SentenceTransformer('clip-ViT-B-32', device='cpu')
    # 这里只是示例，实际应用中需要加载真实图片
    # 为了演示，我们创建一些空白图像
    from PIL import Image
    blank_images = [Image.new('RGB', (224, 224), color=(255, 255, 255)) for _ in range(len(df))]
    img_emb = img_model.encode(blank_images, batch_size=16, show_progress_bar=True)
    print(f"图像特征维度: {img_emb.shape}")
except Exception as e:
    print(f"图像模型加载失败，使用零向量替代: {e}")
    img_emb = np.zeros((len(df), 512))  # CLIP模型输出维度为512

# 数值特征
num_feats = df[['ctr', 'impr_1h', 'adv_hist_violation']].values

# 特征拼接
X = np.hstack([text_emb, img_emb, num_feats])
y = df['label'].values

# ========== 5. 训练集/验证集划分 ==========
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ========== 6. 不平衡处理 + 模型训练 ==========
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import joblib
import time
import os

# 创建模型保存目录
os.makedirs('models', exist_ok=True)

# 使用SVD降维后进行SMOTE过采样
svd = TruncatedSVD(n_components=64, random_state=42)
X_train_svd = svd.fit_transform(X_train)

# SMOTE过采样
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_svd, y_train)

# 实时可视化的回调函数
class VisualizationCallback:
    """LightGBM训练可视化回调函数"""
    def __init__(self, X_val, y_val, update_freq=10):
        self.X_val = X_val
        self.y_val = y_val
        self.update_freq = update_freq
        self.iteration = []
        self.losses = []
        self.aucs = []
        self.aps = []
        
        # 设置中文字体，解决中文显示问题
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']  # 优先使用的中文字体列表
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        # 创建绘图
        self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 5))
        self.axes[0].set_title('损失曲线')
        self.axes[0].set_xlabel('迭代次数')
        self.axes[0].set_ylabel('损失值')
        self.axes[1].set_title('AUC曲线')
        self.axes[1].set_xlabel('迭代次数')
        self.axes[1].set_ylabel('AUC值')
        self.axes[2].set_title('PR曲线')
        self.axes[2].set_xlabel('召回率')
        self.axes[2].set_ylabel('精确率')
        plt.ion()  # 开启交互模式
        self.fig.show()
        self.start_time = time.time()
        
    def __call__(self, env):
        i = env.iteration
        if i % self.update_freq != 0 and i != env.end_iteration - 1:
            return
            
        self.iteration.append(i)
        
        # 获取评估结果
        if env.evaluation_result_list:
            # LightGBM的评估结果格式是 (dataset_name, metric_name, score, is_higher_better)
            # 我们只关心验证集的AUC
            for eval_result in env.evaluation_result_list:
                dataset_name, metric_name, score, _ = eval_result
                if dataset_name == 'valid_0' and metric_name == 'auc':
                    self.losses.append(-score if score < 0 else 1 - score)  # 转换为损失
                    
            # 计算AUC和AP
            y_pred = env.model.predict(self.X_val)
            auc = roc_auc_score(self.y_val, y_pred)
            ap = average_precision_score(self.y_val, y_pred)
            self.aucs.append(auc)
            self.aps.append(ap)
            
            # 绘制PR曲线
            precision, recall, _ = precision_recall_curve(self.y_val, y_pred)
            self.axes[2].clear()
            self.axes[2].set_title('PR曲线')
            self.axes[2].set_xlabel('召回率')
            self.axes[2].set_ylabel('精确率')
            self.axes[2].plot(recall, precision)
            self.axes[2].grid(True)
                
        # 绘制损失和AUC曲线
        self.axes[0].clear()
        self.axes[1].clear()
        self.axes[0].set_title('损失曲线')
        self.axes[0].set_xlabel('迭代次数')
        self.axes[0].set_ylabel('损失值')
        self.axes[1].set_title('AUC曲线')
        self.axes[1].set_xlabel('迭代次数')
        self.axes[1].set_ylabel('AUC值')
        
        self.axes[0].plot(self.iteration, self.losses, label='验证集')
        self.axes[1].plot(self.iteration, self.aucs, label='验证集')
            
        self.axes[0].legend()
        self.axes[1].legend()
        self.axes[0].grid(True)
        self.axes[1].grid(True)
        
        # 显示当前指标
        elapsed = time.time() - self.start_time
        status_text = f"迭代: {i}, 耗时: {elapsed:.1f}秒\n"
        if len(self.losses) > 0:
            status_text += f"损失: {self.losses[-1]:.4f}, AUC: {self.aucs[-1]:.4f}, AP: {self.aps[-1]:.4f}\n"
        print(status_text)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
        
        # 保存最终的图像
        if i == env.end_iteration - 1:
            plt.savefig('models/training_progress.png')
            
    def finalize(self):
        plt.ioff()
        plt.close()

# 检查是否存在已有模型进行微调
model_path = 'models/ad_audit_model.pkl'
try:
    print("检查是否存在已有模型...")
    model_data = joblib.load(model_path)
    print("找到已有模型，将进行微调")
    clf = model_data['clf']
    svd = model_data['svd']
    original_clf = joblib.load(model_path)['clf']  # 保存原始模型用于比较
    
    # 使用已有的SVD转换训练数据
    X_train_svd = svd.transform(X_train)
    
    # SMOTE过采样
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_svd, y_train)
    
    # 微调参数设置 - 使用学习率衰减和更优化的参数
    n_new_trees = 100
    init_lr = 0.01
    
    # 使用学习率衰减策略
    def lr_decay(current_iter):
        return init_lr * (0.95 ** (current_iter // 10))
    
    clf.set_params(
        learning_rate=init_lr,
        n_estimators=clf.n_estimators + n_new_trees,
        min_child_samples=10,  # 防止过拟合
        reg_alpha=0.01,        # L1正则化
        reg_lambda=0.05,       # L2正则化
        subsample=0.8,         # 随机采样
        colsample_bytree=0.8   # 特征采样
    )
    print("使用优化的学习率和正则化参数进行微调...")
except Exception as e:
    print(f"未找到有效的模型文件或加载失败: {e}")
    print("将训练新模型...")
    # 训练新模型
    clf = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        min_child_samples=20,
        reg_alpha=0.01,
        reg_lambda=0.05,
        subsample=0.8,
        colsample_bytree=0.8
    )
    original_clf = None
    X_train_res, y_train_res = sm.fit_resample(X_train_svd, y_train)

# 设置可视化回调
vis_callback = VisualizationCallback(
    X_val=svd.transform(X_val),
    y_val=y_val,
    update_freq=5
)

# 训练/微调模型
clf.fit(
    X_train_res, y_train_res, 
    eval_set=[(svd.transform(X_val), y_val)], 
    eval_metric='auc', 
    callbacks=[lgb.early_stopping(stopping_rounds=20), vis_callback]
)

# 关闭可视化
vis_callback.finalize()

# 在验证集上评估
probs_val = clf.predict_proba(svd.transform(X_val))[:,1]
print("AP:", average_precision_score(y_val, probs_val), "AUC:", roc_auc_score(y_val, probs_val))

# 模型性能评估，比较微调前后的效果
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

def evaluate_model_performance(model, X, y, model_name="模型"):
    """评估模型性能并打印结果"""
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # 计算各项指标
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc_score = roc_auc_score(y, y_pred_proba)
    
    # 计算PR曲线下面积
    precision_curve, recall_curve, _ = precision_recall_curve(y, y_pred_proba)
    pr_auc = auc(recall_curve, precision_curve)
    
    print(f"\n{model_name}性能评估:")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"ROC AUC: {auc_score:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': auc_score,
        'pr_auc': pr_auc
    }

# 如果是微调模型，比较微调前后的性能
if original_clf is not None:
    print("\n===== 模型微调前后性能比较 =====")
    print("在验证集上评估原始模型和微调后的模型性能...")
    
    # 评估原始模型
    original_metrics = evaluate_model_performance(original_clf, svd.transform(X_val), y_val, "原始模型")
    
    # 评估微调后的模型
    new_metrics = evaluate_model_performance(clf, svd.transform(X_val), y_val, "微调后模型")
    
    # 计算性能提升百分比
    improvement = {}
    for metric in original_metrics:
        change = new_metrics[metric] - original_metrics[metric]
        percent = (change / original_metrics[metric]) * 100 if original_metrics[metric] != 0 else float('inf')
        improvement[metric] = percent
    
    print("\n性能提升百分比:")
    for metric, percent in improvement.items():
        print(f"{metric}: {percent:+.2f}%")
    
    # 绘制ROC曲线比较
    plt.figure(figsize=(10, 8))
    
    # 原始模型ROC曲线
    y_pred_proba_orig = original_clf.predict_proba(svd.transform(X_val))[:, 1]
    fpr_orig, tpr_orig, _ = roc_curve(y_val, y_pred_proba_orig)
    plt.plot(fpr_orig, tpr_orig, label=f'原始模型 (AUC = {original_metrics["roc_auc"]:.4f})')
    
    # 微调后模型ROC曲线
    fpr_new, tpr_new, _ = roc_curve(y_val, probs_val)
    plt.plot(fpr_new, tpr_new, label=f'微调后模型 (AUC = {new_metrics["roc_auc"]:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('假阳性率 (FPR)')
    plt.ylabel('真阳性率 (TPR)')
    plt.title('微调前后ROC曲线比较')
    plt.legend()
    plt.grid(True)
    plt.savefig('models/roc_comparison.png')
    
    # 绘制PR曲线比较
    plt.figure(figsize=(10, 8))
    
    # 原始模型PR曲线
    precision_orig, recall_orig, _ = precision_recall_curve(y_val, y_pred_proba_orig)
    plt.plot(recall_orig, precision_orig, label=f'原始模型 (PR AUC = {original_metrics["pr_auc"]:.4f})')
    
    # 微调后模型PR曲线
    precision_new, recall_new, _ = precision_recall_curve(y_val, probs_val)
    plt.plot(recall_new, precision_new, label=f'微调后模型 (PR AUC = {new_metrics["pr_auc"]:.4f})')
    
    plt.xlabel('召回率 (Recall)')
    plt.ylabel('精确率 (Precision)')
    plt.title('微调前后PR曲线比较')
    plt.legend()
    plt.grid(True)
    plt.savefig('models/pr_comparison.png')
    print("已保存ROC和PR曲线比较图到models目录")
else:
    # 仅评估新训练的模型
    evaluate_model_performance(clf, svd.transform(X_val), y_val, "新训练模型")

# ========== 7. PR 曲线阈值选择 ==========
from sklearn.metrics import precision_recall_curve, confusion_matrix
precision, recall, thresholds = precision_recall_curve(y_val, probs_val)

# 业务成本设置
C_FP, C_FN = 1.0, 50.0  # 误判成本和漏判成本
best_t, best_cost = 0.5, float('inf')

# 基于成本的阈值选择
for t in np.linspace(0.01, 0.99, 50):
    y_pred = (probs_val >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    cost = C_FP*fp + C_FN*fn
    if cost < best_cost:
        best_cost, best_t = cost, t
print("Best threshold:", best_t, "with cost:", best_cost)

# 另一种做法：选择使 recall >= target 而 precision 尽可能高
target_recall = 0.8
candidates = [(p,r,t) for p,r,t in zip(precision, recall, list(thresholds)+[1.0]) if r>=target_recall]
if candidates:
    best = max(candidates, key=lambda x: x[0])  # 选择满足recall要求下precision最高的阈值
    print(f"Threshold for recall>={target_recall:.2f} is {best[2]:.3f} (precision={best[0]:.3f})")

# 创建特征语义映射
feature_semantics = {
    # 文本特征
    "text_0": "广告文本-主题相关性",
    "text_1": "广告文本-情感倾向",
    "text_2": "广告文本-紧迫感",
    # 图像特征
    "img_0": "图像-色彩饱和度",
    "img_1": "图像-人脸检测",
    "img_2": "图像-文字占比",
    # 数值特征
    "num_0": "点击率(CTR)",
    "num_1": "小时展示量",
    "num_2": "历史违规次数"
}

# 保存模型和最佳阈值
os.makedirs('models', exist_ok=True)
model_data = {
    'clf': clf,
    'svd': svd,
    'best_threshold': best_t,
    'best_threshold_cost': best_cost,
    'target_recall_threshold': best[2] if candidates else 0.5,
    'feature_names': [f"feature_{i}" for i in range(svd.transform(X_val).shape[1])],
    'feature_semantics': feature_semantics  # 存储特征语义映射
}
joblib.dump(model_data, 'models/ad_audit_model.pkl')

# ========== 8. SHAP 模型解释 ==========
try:
    import shap, matplotlib.pyplot as plt
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(svd.transform(X_val))
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # 全局特征重要性
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[-10:][::-1]
    print("Top contributing features:")
    for idx in top_idx:
        feature_name = f"feature_{idx}"
        # 尝试获取特征语义
        semantic = feature_semantics.get(feature_name, "")
        if semantic:
            print(f"{feature_name} ({semantic}), mean_abs_shap={mean_abs_shap[idx]:.4f}")
        else:
            print(f"{feature_name}, mean_abs_shap={mean_abs_shap[idx]:.4f}")
    
    # 创建带有语义的特征名称列表
    feature_names_with_semantics = []
    for i in range(svd.transform(X_val).shape[1]):
        feature_name = f"feature_{i}"
        semantic = feature_semantics.get(feature_name, "")
        if semantic:
            feature_names_with_semantics.append(f"{feature_name} ({semantic})")
        else:
            feature_names_with_semantics.append(feature_name)
    
    # 保存SHAP摘要图，使用带语义的特征名称
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values, 
        svd.transform(X_val), 
        feature_names=feature_names_with_semantics,
        show=False
    )
    plt.tight_layout()
    plt.savefig("models/shap_summary.png", dpi=300, bbox_inches='tight')
    print("Saved SHAP summary plot to models/shap_summary.png")
except Exception as e:
    print("SHAP failed:", e)

# ========== 9. 端到端广告审核功能 ==========
def process_image_url(url):
    """处理图片URL，提取OCR文本并进行规则检查"""
    try:
        # 下载并预处理图片
        img = download_image(url)
        img_pre = preprocess_for_ocr(img)
        ocr_text = ocr_image(img_pre)
        
        # 规则检查
        blocked, reason = rule_check_text(ocr_text)
        return {
            "ocr_text": ocr_text,
            "rule_block": blocked,
            "rule_reason": reason
        }
    except Exception as e:
        print(f"图片处理失败: {e}")
        return {"ocr_text": "", "rule_block": False, "rule_reason": None}

def predict_ad_with_image(ad_text="", image_url=None, ctr=0.05, impr_1h=10, adv_hist_violation=0):
    """完整的广告审核功能，支持文本和图片输入"""
    result = {"block": False, "reasons": [], "details": {}}
    
    # 1. 文本规则检查
    if ad_text:
        rule_block, reason = rule_check_text(ad_text)
        if rule_block:
            result["block"] = True
            result["reasons"].append(f"文本规则违规: {reason}")
            result["details"]["text_rule_violation"] = reason
    
    # 2. 图片OCR和规则检查
    ocr_text = ""
    if image_url:
        img_result = process_image_url(image_url)
        ocr_text = img_result.get("ocr_text", "")
        if img_result.get("rule_block"):
            result["block"] = True
            result["reasons"].append(f"图片OCR规则违规: {img_result.get('rule_reason')}")
            result["details"]["image_rule_violation"] = img_result.get("rule_reason")
    
    # 如果规则层已经拦截，直接返回结果
    if result["block"]:
        return result
    
    # 3. 模型预测
    try:
        # 特征提取
        doc = (ad_text or "") + " " + (ocr_text or "")
        text_feature = text_model.encode([doc])
        
        # 图像特征
        img_feature = np.zeros((1, 512))  # 默认空特征
        if image_url:
            try:
                img = download_image(image_url)
                if img and img_model:
                    img_feature = img_model.encode([img])
            except Exception as e:
                print(f"图像特征提取失败: {e}")
        
        # 数值特征
        num_feature = np.array([[ctr, impr_1h, adv_hist_violation]])
        
        # 特征拼接
        features = np.hstack([text_feature, img_feature, num_feature])
        
        # 加载模型和最佳阈值
        model_data = joblib.load('models/ad_audit_model.pkl')
        clf = model_data['clf']
        svd = model_data['svd']
        best_threshold = model_data.get('best_threshold', 0.5)  # 使用保存的最佳阈值，如果没有则默认0.5
        
        # 模型预测
        prob = clf.predict_proba(svd.transform(features))[:,1][0]
        decision = (prob >= best_threshold)
        
        # 添加预测结果
        result["details"]["model_probability"] = float(prob)
        result["details"]["model_threshold"] = float(best_t)
        
        if decision:
            result["block"] = True
            result["reasons"].append(f"模型预测违规 (概率: {prob:.4f})")
            
            # 添加SHAP解释
            try:
                feature_svd = svd.transform(features)
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer.shap_values(feature_svd)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1][0]
                else:
                    shap_values = shap_values[0]
                
                # 获取最重要的特征
                top_features = np.argsort(np.abs(shap_values))[-5:][::-1]
                result["details"]["top_violation_features"] = [
                    {"feature_id": int(idx), "importance": float(shap_values[idx])}
                    for idx in top_features
                ]
            except Exception as e:
                print(f"SHAP解释失败: {e}")
    
    except Exception as e:
        print(f"模型预测失败: {e}")
        result["details"]["model_error"] = str(e)
    
    return result

# ========== 10. 测试接口示例 ==========
print("\n===== 测试端到端广告审核功能 =====")
print("测试1 - 文本违规:", predict_ad_with_image("Free iPhone giveaway"))
print("测试2 - 正常广告:", predict_ad_with_image("Educational AI course for beginners"))
print("测试3 - 带图片的广告 (模拟):", predict_ad_with_image(
    "Check out our new products", 
    image_url="https://picx.zhimg.com/v2-e78b138371c3f999fbf589184a442aa6_720w.jpg?source=172ae18b"  # 这里需要替换为真实图片URL进行测试
))

# 保存模型和预处理器
import joblib
try:
    os.makedirs("models", exist_ok=True)
    joblib.dump({
        'svd': svd, 
        'clf': clf, 
        'text_model_name': 'all-MiniLM-L6-v2',
        'img_model_name': 'clip-ViT-B-32',
        'threshold': best_t
    }, "models/ad_audit_model.pkl")
    print("模型已保存到 models/ad_audit_model.pkl")
except Exception as e:
    print(f"模型保存失败: {e}")

print("\n广告审核系统初始化完成！")
