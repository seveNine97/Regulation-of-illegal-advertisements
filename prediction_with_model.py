import pickle
import os
import numpy as np
import argparse
import requests
from PIL import Image
import io
import re
import warnings
import sys
import json
import shap

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning)

# 特征语义映射字典 - 用于SHAP解释
FEATURE_SEMANTICS = {
    # 文本特征语义映射
    "text_length": "文本长度",
    "free_keyword": "免费/free关键词",
    "limited_keyword": "限时/limited关键词",
    "guarantee_keyword": "保证/guarantee关键词",
    "discount_keyword": "折扣/discount关键词",
    "money_keyword": "赚钱/致富关键词",
    "lottery_keyword": "中奖/彩票关键词",
    "loan_keyword": "贷款/借钱关键词",
    "adult_keyword": "成人/情趣关键词",
    "has_image": "包含图片",
    "ctr": "点击率"
}

# 违禁词配置文件路径
FORBIDDEN_WORDS_PATH = os.path.join('models', 'forbidden_words.json')

# 加载违禁词配置
def load_forbidden_words():
    """加载违禁词配置，如果文件不存在则创建默认配置"""
    if not os.path.exists(FORBIDDEN_WORDS_PATH):
        # 默认违禁词配置
        default_config = {
            "exact_match": [
                "100%中奖", "免费", "贷款无利息", "零首付", "一夜暴富", 
                "无抵押贷款", "免费领取", "免费试用", "秒批", "秒过", "包过"
            ],
            "regex_patterns": [
                r"free", r"get rich", r"贷款.*无利息", r"zero\s*fee",
                r"\d+\s*%\s*中奖", r"免费\s*领", r"免费\s*试用"
            ],
            "fuzzy_keywords": [
                "free", "iPhone", "中奖", "返现", "暴富", "贷款", "博彩"
            ],
            "fuzzy_threshold": 85
        }
        
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(FORBIDDEN_WORDS_PATH), exist_ok=True)
        
        # 保存默认配置
        with open(FORBIDDEN_WORDS_PATH, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
        
        return default_config
    
    # 加载现有配置
    try:
        with open(FORBIDDEN_WORDS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载违禁词配置失败: {e}，使用默认配置")
        return {
            "exact_match": ["免费", "中奖"],
            "regex_patterns": [r"free", r"get rich"],
            "fuzzy_keywords": ["free", "中奖"],
            "fuzzy_threshold": 85
        }

# 更新违禁词配置
def update_forbidden_words(new_words=None, new_patterns=None, new_fuzzy=None, fuzzy_threshold=None):
    """更新违禁词配置"""
    config = load_forbidden_words()
    
    if new_words:
        for word in new_words:
            if word not in config["exact_match"]:
                config["exact_match"].append(word)
    
    if new_patterns:
        for pattern in new_patterns:
            if pattern not in config["regex_patterns"]:
                config["regex_patterns"].append(pattern)
    
    if new_fuzzy:
        for keyword in new_fuzzy:
            if keyword not in config["fuzzy_keywords"]:
                config["fuzzy_keywords"].append(keyword)
    
    if fuzzy_threshold is not None:
        config["fuzzy_threshold"] = fuzzy_threshold
    
    # 保存更新后的配置
    with open(FORBIDDEN_WORDS_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    
    return config

# 规则检查函数
def rule_check_text(text):
    """对文本进行规则检查，使用动态加载的违禁词配置"""
    if text is None or text == "":
        return False, []
    
    # 加载违禁词配置
    config = load_forbidden_words()
    
    violations = []
    
    # 精确匹配
    for word in config["exact_match"]:
        if word in text:
            violations.append(f"exact:{word}")
    
    # 正则匹配
    for pattern in config["regex_patterns"]:
        if re.search(pattern, text, flags=re.I):
            violations.append(f"regex:{pattern}")
    
    # 模糊匹配
    try:
        from rapidfuzz import fuzz
        threshold = config.get("fuzzy_threshold", 85)
        
        for keyword in config["fuzzy_keywords"]:
            score = fuzz.partial_ratio(keyword, text)
            if score > threshold:
                violations.append(f"fuzzy:{keyword}:{score}")
    except ImportError:
        print("警告: rapidfuzz库未安装，跳过模糊匹配")
    
    return len(violations) > 0, violations

# 图片处理和OCR函数
def download_image(image_url):
    """下载图片"""
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    except Exception as e:
        print(f"图片下载失败: {e}")
        return None

def preprocess_for_ocr(image):
    """预处理图片以提高OCR效果"""
    if image is None:
        return None
    
    # 转为灰度图
    try:
        gray = image.convert('L')
        # 可以添加更多预处理步骤，如二值化、去噪等
        return gray
    except Exception as e:
        print(f"图片预处理失败: {e}")
        return image

def ocr_image(image):
    """OCR提取图片文本"""
    if image is None:
        return ""
    
    try:
        import pytesseract
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"OCR处理警告: {e}")
        print("提示: 请安装Tesseract OCR (https://github.com/UB-Mannheim/tesseract/wiki)")
        return ""

# 加载模型和预处理器
def load_model():
    """加载保存的模型和预处理器"""
    model_path = os.path.join('models', 'ad_audit_model.pkl')
    try:
        print("加载模型文件...")
        # 使用joblib而不是pickle来加载模型
        import joblib
        model_data = joblib.load(model_path)
        
        print("模型加载成功")
        return model_data
    except Exception as e:
        print(f"模型加载失败: {e}")
        sys.exit(1)

# 特征向量到语义的映射函数
def map_feature_to_semantic(feature_idx, svd_components, top_n=5):
    """将SVD转换后的特征映射回原始特征空间，并返回语义解释"""
    # 获取该特征在SVD组件中的权重
    feature_weights = np.abs(svd_components[feature_idx])
    
    # 获取权重最高的几个原始特征索引
    top_indices = np.argsort(feature_weights)[-top_n:][::-1]
    
    # 映射到语义
    semantics = []
    for idx in top_indices:
        # 为简化示例，这里使用固定映射
        # 实际应用中，可以根据特征索引范围判断特征类型
        if idx < 384:  # 假设前384维是文本特征
            semantic = "文本特征"
            if idx < 10:
                semantic = FEATURE_SEMANTICS.get(f"text_feature_{idx}", semantic)
        elif idx < 896:  # 假设384-896是图像特征
            semantic = "图像特征"
        else:  # 剩余是数值特征
            feature_name = ""
            if idx == 896:
                feature_name = "text_length"
            elif idx == 897:
                feature_name = "has_image"
            elif idx == 898:
                feature_name = "ctr"
            
            semantic = FEATURE_SEMANTICS.get(feature_name, f"数值特征_{idx-896}")
        
        semantics.append((semantic, feature_weights[idx]))
    
    return semantics
def predict_ad(text=None, image_url=None):
    """预测广告是否违规"""
    # 加载模型数据
    model_data = load_model()
    
    # 解包模型组件
    clf = model_data['clf']    # LightGBM模型
    svd = model_data['svd']    # 降维组件
    threshold = model_data.get('threshold', 0.39)  # 最佳阈值，默认0.39
    
    # 打印模型数据键，帮助调试
    print(f"模型数据包含以下键: {list(model_data.keys())}")
    
    # 步骤1: 规则检查
    rule_violation, rule_reasons = False, []
    if text:
        rule_violation, rule_reasons = rule_check_text(text)
        if rule_violation:
            return {
                'block': True,
                'reasons': [f'文本规则违规: {reason}' for reason in rule_reasons],
                'details': {
                    'text_rule_violation': ', '.join(rule_reasons),
                    'violation_type': '规则违规',
                    'matched_words': rule_reasons,
                    'text_snippet': text[:100] + ('...' if len(text) > 100 else '')
                }
            }
    
    # 步骤2: 如果有图片，尝试OCR
    ocr_text = ""
    if image_url:
        image = download_image(image_url)
        if image:
            processed_image = preprocess_for_ocr(image)
            ocr_text = ocr_image(processed_image)
            # 对OCR文本进行规则检查
            if ocr_text:
                ocr_violation, ocr_reasons = rule_check_text(ocr_text)
                if ocr_violation:
                    return {
                        'block': True,
                        'reasons': [f'图片文本规则违规: {reason}' for reason in ocr_reasons],
                        'details': {
                            'ocr_text': ocr_text, 
                            'ocr_rule_violation': ', '.join(ocr_reasons),
                            'violation_type': '图片文字违规',
                            'matched_words': ocr_reasons
                        }
                    }
    
    # 步骤3: 使用模型中保存的特征向量
    # 这里我们假设模型中已经包含了所有需要的特征向量
    # 我们只需要创建一个简单的特征向量，让模型能够进行预测
    
    # 获取SVD的输入维度
    svd_input_dim = svd.n_features_in_
    print(f"SVD期望的输入维度: {svd_input_dim}")
    
    # 创建一个与SVD输入维度匹配的特征向量
    features = np.zeros(svd_input_dim)
    
    # 如果有文本，设置一些基本特征
    if text:
        # 设置文本长度特征（放在倒数第三个位置）
        features[-3] = len(text)
        # 设置一些关键词特征
        if "免费" in text or "free" in text.lower():
            features[-5] = 1
        if "限时" in text or "limited" in text.lower():
            features[-6] = 1
    
    # 如果有图片，设置图片特征标志（放在倒数第二个位置）
    if image_url:
        features[-2] = 1
        
    # 设置一个默认的数值特征（放在最后一个位置）
    features[-1] = 0.05  # 默认CTR值
    
    # 步骤4: 使用模型预测
    try:
        # 转换特征并预测
        features_transformed = svd.transform([features])
        prediction_prob = clf.predict_proba(features_transformed)[0, 1]
        is_violation = prediction_prob >= threshold
        
        # 返回结果
        if is_violation:
            # 使用SHAP解释结果
            try:
                # 创建SHAP解释器
                explainer = shap.Explainer(clf, svd.transform(np.zeros((1, svd_input_dim))))
                # 计算SHAP值
                shap_values = explainer(features_transformed)
                
                # 获取最重要的特征索引
                feature_importance = shap_values.values[0, :, 1]  # 第二个类别是违规类
                top_feature_indices = np.argsort(-np.abs(feature_importance))[:5]  # 取前5个最重要的特征
                
                # 获取特征语义解释
                violation_reasons = []
                for idx in top_feature_indices:
                    if feature_importance[idx] > 0:  # 只关注正向贡献
                        semantics = map_feature_to_semantic(idx, svd.components_, top_n=3)
                        for semantic, weight in semantics:
                            violation_reasons.append({
                                'feature': semantic,
                                'importance': float(weight),
                                'contribution': float(feature_importance[idx])
                            })
                
                risk_factors = [r['feature'] for r in violation_reasons[:3]]
                detailed_reason = f"模型预测为违规广告，主要风险因素: {', '.join(risk_factors)}"
                
                return {
                    'block': True,
                    'reasons': [detailed_reason],
                    'details': {
                        'model_probability': float(prediction_prob),
                        'model_threshold': threshold,
                        'violation_type': '模型预测违规',
                        'top_reasons': violation_reasons,
                        'risk_factors': risk_factors
                    }
                }
            except Exception as shap_error:
                print(f"SHAP解释生成失败: {shap_error}")
                return {
                    'block': True,
                    'reasons': [f'模型预测为违规广告 (风险分数: {prediction_prob:.2f})'],
                    'details': {
                        'model_probability': float(prediction_prob),
                        'model_threshold': threshold
                    }
                }
        else:
            return {
                'block': False,
                'reasons': [],
                'details': {
                    'model_probability': float(prediction_prob),
                    'model_threshold': threshold
                }
            }
    except Exception as e:
        print(f"预测失败: {e}")
        # 如果预测失败，返回基于规则的结果
        return {
            'block': False,
            'reasons': [],
            'details': {
                'error': str(e),
                'fallback': '模型预测失败，使用规则检查结果'
            }
        }

# 命令行接口
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='广告违规检测工具')
    parser.add_argument('--text', type=str, help='广告文本')
    parser.add_argument('--image', type=str, help='广告图片URL')
    
    args = parser.parse_args()
    
    if not args.text and not args.image:
        print("错误: 请提供广告文本或图片URL")
        parser.print_help()
        sys.exit(1)
    
    result = predict_ad(text=args.text, image_url=args.image)
    print("\n===== 广告审核结果 =====")
    print(f"违规: {'是' if result['block'] else '否'}")
    
    if result['block']:
        print("\n违规原因:")
        for reason in result['reasons']:
            print(f"- {reason}")
    
    print("\n详细信息:")
    for key, value in result['details'].items():
        print(f"- {key}: {value}")