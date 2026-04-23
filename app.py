import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

# --- 1. 页面配置 ---
st.set_page_config(page_title="图传播仿真平台", layout="wide")

# --- 2. 核心网络生成函数 ---
@st.cache_resource
def generate_base_network(n, m, seed):
    np.random.seed(seed)
    g = nx.barabasi_albert_graph(n, m, seed=seed)
    pos = nx.spring_layout(g, k=0.15, iterations=20, seed=seed)
    return g, pos

# --- 3. 侧边栏：参数控制 ---
st.sidebar.header("⚙️ 实验参数设置")
n_slider = st.sidebar.slider("节点数量 (N)", 200, 2000, 800)
m_slider = st.sidebar.slider("网络稠密度 (M)", 1, 5, 2)
t_slider = st.sidebar.slider("传播概率 (T)", 0.05, 0.4, 0.15)
alpha_slider = st.sidebar.slider("风险阈值 (α)", 0.05, 0.3, 0.12)
hunt_slider = st.sidebar.slider("开始干预步数", 1, 15, 5)
det_slider = st.sidebar.slider("流调检测率", 0.1, 1.0, 0.7)
seed_input = st.sidebar.number_input("随机种子 (Seed)", value=42)

# --- 4. Session State 逻辑自检与初始化 ---
# 如果核心结构参数(N, M, Seed)改变，强制重置所有状态
param_fingerprint = f"{n_slider}_{m_slider}_{seed_input}"

if 'fingerprint' not in st.session_state or st.session_state.fingerprint != param_fingerprint or st.sidebar.button("🔄 强制重置模拟"):
    st.session_state.fingerprint = param_fingerprint
    st.session_state.step = 0
    st.session_state.running = False
    
    # 重新生成图
    g, pos = generate_base_network(n_slider, m_slider, seed_input)
    st.session_state.g = g
    st.session_state.pos = pos
    
    # 重新初始化疫情数据
    st.session_state.true_source = np.random.choice(list(g.nodes()))
    st.session_state.infected = {st.session_state.true_source}
    st.session_state.quarantined = set()
    st.session_state.known_cases = set()
    st.session_state.pred_source = None
    st.session_state.is_contained = False

# 实时更新边权重（T改变无需重绘图，但需更新逻辑）
weight_val = -np.log(max(t_slider, 1e-5))
for u, v in st.session_state.g.edges():
    st.session_state.g[u][v]['weight'] = weight_val

# --- 5. UI 头部展示（修复颜色与数据滞后问题） ---
st.title("🛡️ 大规模图传播动态溯源、隔离与风险评估系统")

# 强制文字为黑色，背景为淡灰色，确保在深色模式下也清晰
actual_n = len(st.session_state.g.nodes())
st.markdown(f"""
<div style="background-color:#f8f9fa; padding:15px; border-radius:10px; margin-bottom:20px; color:#000000; border: 2px solid #dee2e6;">
    <p style="margin:0;"><b>当前网络规模：</b> {actual_n} 节点 | <b>传播概率：</b> {t_slider} | <b>监测率：</b> {det_slider}</p>
    <p style="margin:0; margin-top:5px;"><b>图例：</b> 🌟 真源头 | ✖️ 预测源头 | <span style="color:red;">●</span> 真实感染 | <span style="color:#6c757d;">●</span> 隔离区 | <span style="color:orange;">●</span> 高风险预警</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

# --- 6. 仿真逻辑 ---
def run_simulation_step():
    s = st.session_state
    if s.is_contained: return

    # A. 传播逻辑：未隔离的感染者向未隔离的健康邻居传播
    active_spreaders = s.infected - s.quarantined
    
    # 终止检查：如果没有可以传播的人了
    if len(active_spreaders) == 0 and s.step > 0:
        s.is_contained = True
        return

    new_infections = set()
    for u in list(active_spreaders):
        for v in s.g.neighbors(u):
            if v not in s.infected and v not in s.quarantined:
                if np.random.rand() < t_slider:
                    new_infections.add(v)
    s.infected.update(new_infections)

    # B. 溯源与隔离逻辑
    if s.step >= hunt_slider:
        # 只能探测到未隔离感染者中的一部分
        newly_found = {n for n in (s.infected - s.quarantined) if np.random.rand() < det_slider}
        s.known_cases.update(newly_found)
        
        if len(s.known_cases) > 1:
            try:
                # 斯坦纳树计算隔离路径
                st_tree = nx.algorithms.approximation.steiner_tree(s.g, list(s.known_cases), weight='weight')
                s.quarantined.update(st_tree.nodes())
                
                # 预测最初源头
                if nx.is_connected(st_tree):
                    sub = st_tree
                else:
                    sub = st_tree.subgraph(max(nx.connected_components(st_tree), key=len))
                ecc = nx.eccentricity(sub)
                s.pred_source = min(ecc, key=ecc.get)
            except: pass
    s.step += 1

# --- 7. 绘图函数 ---
def render_map():
    s = st.session_state
    fig, ax = plt.subplots(figsize=(10, 7), dpi=100)
    
    all_nodes = set(s.g.nodes())
    tp_nodes = s.infected & s.quarantined
    fp_nodes = s.quarantined - s.infected
    fn_nodes = s.infected - s.quarantined
    tn_nodes = all_nodes - s.infected - s.quarantined
    
    # 计算风险点 (属于 TN)
    high_risk = set()
    for v in tn_nodes:
        leaked_neighs = [n for n in s.g.neighbors(v) if n in fn_nodes]
        if (1 - (1 - t_slider)**len(leaked_neighs)) > alpha_slider:
            high_risk.add(v)

    # 绘图层
    nx.draw_networkx_edges(s.g, s.pos, ax=ax, alpha=0.03, edge_color='#cccccc')
    
    # 1. 基础健康节点 (蓝色)
    nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(tn_nodes - high_risk), node_color='#87CEEB', node_size=30, ax=ax)
    # 2. 漏网者 (红色黑边)
    nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(fn_nodes), node_color='red', node_size=90, edgecolors='black', ax=ax)
    # 3. 隔离成功 (灰色底+红色心)
    if tp_nodes:
        nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(tp_nodes), node_color='#A9A9A9', node_size=120, ax=ax)
        nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(tp_nodes), node_color='red', node_size=25, ax=ax)
    # 4. 误伤 (纯灰色)
    if fp_nodes:
        nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(fp_nodes), node_color='#D3D3D3', node_size=60, ax=ax)
    # 5. 高风险 (橙色)
    if high_risk:
        nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(high_risk), node_color='orange', node_size=80, ax=ax)

    # 特殊标识
    ax.scatter(s.pos[s.true_source][0], s.pos[s.true_source][1], marker='*', s=450, c='gold', edgecolors='black', zorder=10)
    if s.pred_source is not None:
        ax.scatter(s.pos[s.pred_source][0], s.pos[s.pred_source][1], marker='x', s=350, c='purple', zorder=11)
    
    ax.axis('off')
    try:
        err = nx.shortest_path_length(s.g, s.true_source, s.pred_source) if s.pred_source is not None else "N/A"
    except: err = "Unreachable"
    
    ax.set_title(f"Step: {s.step} | Error Distance: {err}", fontsize=15)
    return fig, len(tp_nodes), len(fp_nodes), len(fn_nodes), len(tn_nodes)

# --- 8. 实时显示控制 ---
plot_area = col1.empty()
table_area = col2.empty()
metrics_area = col2.empty()

if st.sidebar.button("▶️ 启动/继续"):
    st.session_state.running = True

if st.session_state.running:
    while not st.session_state.is_contained:
        run_simulation_step()
        fig, tp, fp, fn, tn = render_map()
        plot_area.pyplot(fig)
        plt.close(fig)

        # 数据校验：tp + fp + fn + tn 必须等于 actual_n
        total_check = tp + fp + fn + tn
        
        with table_area.container():
            st.write("#### 📊 2x2 混淆矩阵")
            # 这里的 index 和 column 名精简以适配布局
            df = pd.DataFrame({
                "Quarantined": [f"TP: {tp}", f"FP: {fp}"],
                "Non-Quarantined": [f"FN: {fn}", f"TN: {tn}"]
            }, index=["Sick (Red)", "Safe (Blue)"])
            st.table(df)
        
        with metrics_area.container():
            acc = (tp + tn) / total_check
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            st.metric("实时准确率 (Accuracy)", f"{acc:.1%}")
            st.metric("查全率 (Recall/Power)", f"{rec:.1%}")
        
        if st.session_state.is_contained:
            st.success(f"🎉 疫情清除！总步数: {st.session_state.step}")
            st.balloons()
            st.session_state.running = False
            break
        time.sleep(0.05)
else:
    # 初始显示
    fig, tp, fp, fn, tn = render_map()
    plot_area.pyplot(fig)
    plt.close(fig)
    st.info("设置好参数后，点击左侧按钮开始。")
