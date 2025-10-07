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

# ========== 1. 模拟数据生成 ==========
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
from sklearn.metrics import average_precision_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import TruncatedSVD

# 使用SVD降维后进行SMOTE过采样
svd = TruncatedSVD(n_components=64, random_state=42)
X_train_svd = svd.fit_transform(X_train)

# SMOTE过采样
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_svd, y_train)

# 训练LightGBM模型
clf = lgb.LGBMClassifier(n_estimators=200)
clf.fit(X_train_res, y_train_res, eval_set=[(svd.transform(X_val), y_val)], eval_metric='auc', callbacks=[lgb.early_stopping(stopping_rounds=20)])

# 在验证集上评估
probs_val = clf.predict_proba(svd.transform(X_val))[:,1]
print("AP:", average_precision_score(y_val, probs_val), "AUC:", roc_auc_score(y_val, probs_val))

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
        print(f"feature_{idx}, mean_abs_shap={mean_abs_shap[idx]:.4f}")
    
    # 保存SHAP摘要图
    shap.summary_plot(shap_values, svd.transform(X_val), show=False)
    plt.savefig("shap_summary.png")
    print("Saved SHAP summary plot to shap_summary.png")
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
        
        # 模型预测
        prob = clf.predict_proba(svd.transform(features))[:,1][0]
        decision = (prob >= best_t)
        
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
