
# streamlit run app.py 运行
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt


@st.cache_resource
def load_resources():
    # 加载7维预处理器和模型
    with open('rf_model_4features.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('preprocessors_4features.pkl', 'rb') as f:
        preprocessors = pickle.load(f)
    explainer = shap.TreeExplainer(model)
    return model, explainer, preprocessors


model, explainer, preprocessors = load_resources()
scaler = preprocessors['scaler']  # 7维标准化器
normalizer = preprocessors['normalizer']  # 7维正则化器

# 用户输入界面
st.title('企业守信预测')
st.markdown("请输入指标：")

# 输入字段
CW02 = st.number_input("资本实力(万)")
CR02 = st.number_input("历史处置记录(次数)")
CP03 = st.number_input("招聘信息岗位对外招收量(个) ")
CR04 = st.number_input("司法案件数(个) ")
CS02 = st.number_input("参保人数(人) ")
CP02 = st.number_input("专利数量(个) ")
CS01 = st.number_input("企业规模(1=微型，2=小型，3=中型，4=大型) ")

if st.button('守信概率'):
    input_data = pd.DataFrame([[CW02, CR02, CP03, CR04, CS02, CP02, CS01]],
                              columns=['CW02', 'CR02', 'CP03', 'CR04', 'CS02', 'CP02', 'CS01'])
    input_scaled = scaler.transform(input_data)
    input_processed = normalizer.transform(input_scaled)

    prob = model.predict_proba(input_processed)[0, 1]
    # 数值显示
    # st.success(f"**守信概率：{prob:.4%}**")

    # 根据概率值划分层次
    if prob < 0.85:
        level = "预警级"
        color = "red"
    elif prob < 0.992:
        level = "审核级"
        color = "orange"
    elif prob < 0.998:
        level = "观察级"
        color = "Yellow"
    else:
        level = "优质级"
        color = "green"

    # 使用HTML标记和颜色显示结果
    # 只显示文字结果
    st.markdown(f"<p style='font-size:20px;'>守信概率：<span style='color:{color};font-weight:bold;'>{level}</span></p>",
                unsafe_allow_html=True)

    with st.expander("点击查看数据处理细节"):
        st.write("原始输入值：", input_data.values)
        st.write("标准化后：", input_scaled)
        st.write("归一化后：", input_processed)

    # 显示原始输入值作为参考
    st.markdown("**当前输入值:**")
    st.dataframe(input_data.style.format("{:.1f}"))