# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 14:27:00 2025

@author: 86185
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

# 设置matplotlib支持中文和负号
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 加载模型
model_path = "RandomForestRegressor.pkl"
model = joblib.load(model_path)

# 设置页面配置和标题
st.set_page_config(layout="wide", page_title="随机森林回归模型预测与 SHAP 可视化", page_icon="📊")
st.title("📊 随机森林回归模型预测与 SHAP 可视化")
st.write("通过输入所有变量的值进行单个样本分娩心理创伤的风险预测，可以得到该样本罹患分娩心理创伤的概率，并结合 SHAP 力图分析结果，有助于临床医护人员了解具体的风险因素和保护因素。")

# 特征范围定义
feature_ranges = {
    "年龄": {"type": "numerical", "min": 18, "max": 42, "default": 18},
    "体重": {"type": "numerical", "min": 52, "max": 91, "default": 52},
    # 添加其他特征...
}

# 动态生成输入项
st.sidebar.header("特征输入区域")
st.sidebar.write("请输入特征值：")

feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.sidebar.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=properties["min"],
            max_value=properties["max"],
            value=properties["default"],
        )
    elif properties["type"] == "categorical":
        value = st.sidebar.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 转换为模型输入格式
features_df = pd.DataFrame([feature_values], columns=list(feature_ranges.keys()))

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    predicted_value = model.predict(features_df)[0]
    st.write(f"Predicted 分娩心理创伤 score: {predicted_value:.2f}")

    # SHAP 解释器
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_df)

    # SHAP 力图
    st.write("### SHAP 力图")
    shap.initjs()
    force_plot = shap.force_plot(explainer.expected_value, shap_values[0, :], features_df.iloc[0, :], feature_names=features_df.columns.tolist())
    st.pyplot(force_plot)

    # 展示蜂群图
    st.write("### 蜂群图")
    image_url = "https://github.com/wuyuze3387/-03.25/blob/main/蜂群图.png"
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # 确保请求成功
        img = Image.open(BytesIO(response.content))
        st.image(img, caption='蜂群图', use_column_width=True)
    except requests.exceptions.RequestException as e:
        st.error("无法加载图片，请检查链接是否正确。错误信息：" + str(e))
