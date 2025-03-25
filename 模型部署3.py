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
    "居住地": {"type": "categorical", "options": [1, 2]},
    "婚姻状况": {"type": "categorical", "options": [1, 2]},
    "就业情况": {"type": "categorical", "options": [1, 2]},
    "学历": {"type": "categorical", "options": [1, 2, 3, 4]},
    "医疗费用支付方式": {"type": "categorical", "options": [1, 2, 3]},
    "怀孕次数": {"type": "numerical", "min": 1, "max": 8, "default": 1},
    "分娩次数": {"type": "numerical", "min": 1, "max": 4, "default": 1},
    "分娩方式": {"type": "categorical", "options": [1, 2, 3]},
    "不良孕产史": {"type": "categorical", "options": [1, 2]},
    "终止妊娠经历": {"type": "categorical", "options": [1, 2]},
    "妊娠周数": {"type": "numerical", "min": 29, "max": 44, "default": 29},
    "妊娠合并症": {"type": "categorical", "options": [1, 2]},
    "妊娠并发症": {"type": "categorical", "options": [1, 2]},
    "喂养方式": {"type": "categorical", "options": [1, 2, 3]},
    "新生儿是否有出生缺陷或疾病": {"type": "categorical", "options": [1, 2]},
    "家庭人均月收入": {"type": "numerical", "min": 1000, "max": 15000, "default": 1000},
    "使用无痛分娩技术": {"type": "categorical", "options": [1, 2]},
    "产时疼痛": {"type": "numerical", "min": 0, "max": 10, "default": 0},
    "产后疼痛": {"type": "numerical", "min": 1, "max": 9, "default": 1},
    "产后照顾婴儿方式": {"type": "categorical", "options": [1, 2, 3, 4, 5]},
    "近1月睡眠质量": {"type": "categorical", "options": [1, 2, 3, 4]},
    "近1月夜间睡眠时长": {"type": "numerical", "min": 3, "max": 11, "default": 3},
    "近1月困倦程度": {"type": "categorical", "options": [1, 2, 3, 4]},
    "孕期体育活动等级": {"type": "categorical", "options": [1, 2, 3, 4]},
    "抑郁": {"type": "numerical", "min": 0, "max": 4, "default": 0},
    "焦虑": {"type": "numerical", "min": 0, "max": 4, "default": 0},
    "侵入性反刍性沉思": {"type": "numerical", "min": 0, "max": 30, "default": 0},
    "目的性反刍性沉思": {"type": "numerical", "min": 0, "max": 28, "default": 0},
    "心理弹性": {"type": "numerical", "min": 6, "max": 30, "default": 6},
    "家庭支持": {"type": "numerical", "min": 0, "max": 10, "default": 0},
}

# 动态生成输入项
st.sidebar.header("特征输入区域")
st.sidebar.write("请输入特征值：")

feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.sidebar.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.sidebar.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    predicted_value = model.predict(features)[0]
    st.write(f"Predicted 分娩心理创伤 score: {predicted_value:.2f}")

    # SHAP 解释器
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    # SHAP 力图
    st.write("### SHAP 力图")
    shap.plots.force(explainer.expected_value, shap_values[0, :], features[0, :], feature_names=list(feature_ranges.keys()))
