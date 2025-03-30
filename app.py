# import os
# # 设置 pycox 数据存储路径
# os.environ['PYCOX_DATA'] = 'models'
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
# import torch
import pickle

st.set_page_config(layout="wide")
# from pycox.models import CoxPH

def data_process(data):
    # 这里假设输入数据顺序为:
    # ['Race', 'Year', 'Age', 'Gender', 'Histological type', 'Primary site',
    #  'Stage', 'Grade', 'Surgery', 'Radiotherapy', 'Chemotherapy',
    #  'Tumor size', 'Number of tumors', 'Tumor extension', 'Laterality',
    #  'Metastasis to bone', 'T stage', 'M stage', 'income',
    #  'Metastasis to brain/liver/lung']
    # 本函数对部分变量进行处理，输出顺序对应模型要求的输入格式
    age, tumor_size, gender, his_type, pri_site, grade, surgery, tumor_ext, distant_met, chem, radio, meta_to_bone, meta_to_bll, m_stage, income = data
    return [0, 1, age, gender, his_type, pri_site, 4, grade, surgery, radio, chem, tumor_size, 0, tumor_ext, 2, meta_to_bone, 3, m_stage, income, meta_to_bll]

def load_setting():
    settings = {
        'Age': {'values': [0, 100], 'type': 'slider', 'init_value': 50, 'add_after': ', year'},
        'Gender': {'values': ["Male", "Female"], 'type': 'selectbox', 'init_value': 0, 'add_after': ''},
        'Primary site': {'values': ["Extremity", "Axial skeleton", "Other"], 'type': 'selectbox', 'init_value': 1, 'add_after': ''},
        'Histological type': {'values': ["Conventional", "Dedifferentiated"], 'type': 'selectbox', 'init_value': 0, 'add_after': ''},
        'Grade': {
            'values': ["Well differentiated", "Moderately differentiated", "Poorly differentiated", "Undifferentiated"],
            'type': 'selectbox', 'init_value': 0, 'add_after': ''},
        'Surgery': {'values': ["None", "Local treatment", "Radical excision with limb salvage", "Amputation"],
                    'type': 'selectbox', 'init_value': 1, 'add_after': ''},
        'Tumor size': {'values': [0, 1000], 'type': 'slider', 'init_value': 135, 'add_after': ', mm'},
        'Tumor extension': {'values': ["No break in periosteum", "Extension beyond periosteum", "Further extension"],
                            'type': 'selectbox', 'init_value': 1, 'add_after': ''},
        'Distant metastasis': {'values': ["Yes", "None"], 'type': 'selectbox',
                               'init_value': 0, 'add_after': ''},
        'Chemotherapy': {'values': ["No", "Yes"], 'type': 'selectbox',
                         'init_value': 0, 'add_after': ''},
        'radiotherapy': {'values': ["No", "Yes"], 'type': 'selectbox',
                         'init_value': 0, 'add_after': ''},
        'metastasis to bone': {'values': ["Unknown", "Yes", "No"], 'type': 'selectbox',
                               'init_value': 0,
                               'add_after': ''},
        'metastasis to brain/liver/lung': {'values': ["Unknown", "Yes", "No"], 'type': 'selectbox',
                                          'init_value': 0,
                                          'add_after': ''},
        'M stage': {'values': ["M0", "M1", "Unknown"], 'type': 'selectbox',
                    'init_value': 2,
                    'add_after': ''},
        'Income': {'values': ["<60,000$", "60,000-80,000$", "80,000-100,000$", ">100,000$"], 'type': 'selectbox',
                   'init_value': 1,
                   'add_after': ''}
    }
    # 仅选择用于预测的输入项
    input_keys = ['Age', 'Tumor size', 'Gender', 'Histological type', 'Primary site', 'Grade', 'Surgery', 'Tumor extension',
                  'Distant metastasis', 'Chemotherapy', 'radiotherapy', 'metastasis to bone', 'metastasis to brain/liver/lung',
                  'M stage', 'Income']
    return settings, input_keys

settings, input_keys = load_setting()

# 使用 session_state 保存患者信息和显示模式
if 'patients' not in st.session_state:
    st.session_state['patients'] = []
if 'display' not in st.session_state:
    st.session_state['display'] = 1
if 'model' not in st.session_state:
    st.session_state['model'] = 'deepsurv'

# 直接在侧边栏构建控件，不使用 exec 动态生成代码
def get_sidebar_controls():
    controls = {}
    for key, setting in settings.items():
        if setting['type'] == 'slider':
            min_val, max_val = setting['values']
            controls[key] = st.slider(key + setting['add_after'], min_val, max_val, setting['init_value'], key=key)
        elif setting['type'] == 'selectbox':
            controls[key] = st.selectbox(key + setting['add_after'], setting['values'], setting['init_value'], key=key)
    return controls

def plot_survival():
    pd_data = pd.concat(
        [
            pd.DataFrame(
                {
                    'Survival': item['survival'],
                    'Time': item['times'],
                    'Patients': [item['No'] for _ in item['times']]
                }
            ) for item in st.session_state['patients']
        ]
    )
    if st.session_state['display']:
        fig = px.line(pd_data, x="Time", y="Survival", color='Patients', range_y=[0, 1])
    else:
        last_patient = pd_data['Patients'].to_list()[-1]
        fig = px.line(pd_data.loc[pd_data['Patients'] == last_patient, :], x="Time", y="Survival", range_y=[0, 1])
    fig.update_layout(template='simple_white',
                      title={
                          'text': 'Estimated Survival Probability',
                          'y': 0.9,
                          'x': 0.5,
                          'xanchor': 'center',
                          'yanchor': 'top',
                          'font': dict(size=25)
                      },
                      plot_bgcolor="white",
                      xaxis_title="Time, month",
                      yaxis_title="Survival probability")
    st.plotly_chart(fig, use_container_width=True)

def plot_patients():
    patients = pd.concat(
        [
            pd.DataFrame(
                dict(
                    {
                        'Patients': [item['No']],
                        '1-Year': ["{:.2f}%".format(item['1-year'] * 100)],
                        '3-Year': ["{:.2f}%".format(item['3-year'] * 100)],
                        '5-Year': ["{:.2f}%".format(item['5-year'] * 100)],
                        '10-Year': ["{:.2f}%".format(item['10-year'] * 100)],
                        'Risk Level': [item['risk level']]
                    },
                    **item['arg']
                )
            ) for item in st.session_state['patients']
        ]
    ).reset_index(drop=True)
    st.dataframe(patients)

def get_model():
    # 这里采用 pickle 加载已经保存好的 CoxPHFitter 模型
    with open("cox_model.pkl", "rb") as f:
        cph_loaded = pickle.load(f)
    return cph_loaded

def scale_result(x):
    x = max(0, x)
    x = min(1, x)
    return x

def predict():
    inputs = []
    for key in input_keys:
        value = st.session_state[key]
        if isinstance(value, int):
            inputs.append(value)
        elif isinstance(value, str):
            inputs.append(settings[key]['values'].index(value))
    
    model = get_model()
    inputs = data_process(inputs)
    inputs = np.expand_dims(np.array(inputs, dtype=np.float32), 0)

    survival = model.predict_cumulative_hazard(inputs)
    survival = np.array(survival)

    # 计算生存概率，假设1-year生存概率 = 1 - 累积风险 at index 12
    one_year = scale_result(1 - survival[12, 0])
    three_year = scale_result(1 - survival[36, 0])
    five_year = scale_result(1 - survival[60, 0])
    ten_year = scale_result(1 - survival[120, 0])
    
    # 根据 1-year 生存概率确定风险等级
    if three_year > 0.85:
        risk_level = "low-risk"
    elif three_year >= 0.7:
        risk_level = "medium-risk"
    else:
        risk_level = "high-risk"
    
    data = {
        'survival': survival.flatten(),
        'times': list(range(0, len(survival.flatten()))),
        'No': len(st.session_state['patients']) + 1,
        'arg': {key: st.session_state[key] for key in input_keys},
        '1-year': one_year,
        '3-year': three_year,
        '5-year': five_year,
        '10-year': ten_year,
        'risk level': risk_level
    }
    st.session_state['patients'].append(data)

def plot_below_header():
    col1, col2 = st.columns([1, 9])
    # 创建5个指标的列：col3 用于 Risk Level，col4~col7 分别用于各生存概率
    col3, col4, col5, col6, col7 = st.columns([2, 2, 2, 2, 2])
    with col1:
        for _ in range(8):
            st.write('')
        st.session_state['display'] = ['Single', 'Multiple'].index(
            st.radio("Display", ('Single', 'Multiple'), st.session_state['display'])
        )
    with col2:
        plot_survival()
    with col3:
        st.metric(
            label='Risk Level',
            value=st.session_state['patients'][-1]['risk level']
        )
    with col4:
        st.metric(
            label='1-Year survival probability',
            value="{:.2f}%".format(st.session_state['patients'][-1]['1-year'] * 100)
        )
    with col5:
        st.metric(
            label='3-Year survival probability',
            value="{:.2f}%".format(st.session_state['patients'][-1]['3-year'] * 100)
        )
    with col6:
        st.metric(
            label='5-Year survival probability',
            value="{:.2f}%".format(st.session_state['patients'][-1]['5-year'] * 100)
        )
    with col7:
        st.metric(
            label='10-Year survival probability',
            value="{:.2f}%".format(st.session_state['patients'][-1]['10-year'] * 100)
        )
    st.write('')
    st.write('')
    st.write('')
    plot_patients()
    for _ in range(4):
        st.write('')

st.header('Data-Driven Prognostic Modeling of Chondrosarcoma Using a Super Learner Ensemble Model', anchor='survival-of-chondrosarcoma')
if st.session_state['patients']:
    plot_below_header()

st.subheader("Instructions:")
st.write("1. Select patient's infomation on the left\n2. Press predict button\n3. The model will generate predictions")
st.write('***Note: this model is still a research subject, and the accuracy of the results cannot be guaranteed!***')
st.write("***[Paper link](https://pubmed.ncbi.nlm.nih.gov/)(To be updated)***")

with st.sidebar:
    with st.form("my_form", clear_on_submit=False):
        sidebar_controls = get_sidebar_controls()
        col8, col9, col10 = st.columns([3, 4, 3])
        with col9:
            prediction = st.form_submit_button('Predict', on_click=predict)
