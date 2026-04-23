import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

# --- 1. 页面配置 ---
st.set_page_config(page_title="大规模图传播与溯源仿真", layout="wide")

# --- 2. 缓存网络生成（节省性能） ---
@st.cache_resource
def get_network(n, m, seed):
    np.random.seed(seed)
    g = nx.barabasi_albert_graph(n, m, seed=seed)
    # 使用 spring 布局
    pos = nx.spring_layout(g, k=0.15, iterations=20, seed=seed)
    return g, pos

# --- 3. 侧边栏：参数控制面板 ---
st.sidebar.header("⚙️ 实验参数设置")
n_val = st.sidebar.slider("节点数量 (N)", 200, 2000, 1000)
m_val = st.sidebar.slider("网络稠密度 (M)", 1, 5, 3)
t_val = st.sidebar.slider("传播概率 (T)", 0.05, 0.4, 0.15)
alpha_val = st.sidebar.slider("风险预警阈值 (α)", 0.05, 0.3, 0.12)
hunt_val = st.sidebar.slider("开始干预步数", 1, 15, 5)
det_val = st.sidebar.slider("流调检测率 (Recall)", 0.1, 1.0, 0.7)
seed_val = st.sidebar.number_input("随机种子 (Seed)", value=42)

# 计算统计学权值
weight_val = -np.log(max(t_val, 1e-5))

# --- 4. 初始化 Session State ---
if 'step' not in st.session_state or st.sidebar.button("🔄 重置实验并运行"):
    st.session_state.step = 0
    st.session_state.running = False
    g, pos = get_network(n_val, m_val, seed_val)
    st.session_state.g = g
    st.session_state.pos = pos
    # 设置权值
    for u, v in st.session_state.g.edges():
        st.session_state.g[u][v]['weight'] = weight_val
    
    st.session_state.true_source = np.random.choice(list(g.nodes()))
    st.session_state.infected = {st.session_state.true_source}
    st.session_state.quarantined = set()
    st.session_state.known_cases = set()
    st.session_state.pred_source = None
    st.session_state.is_contained = False

# --- 5. 主界面标题与图例 ---
st.title("🛡️ 大规模图传播动态溯源、隔离与风险评估系统")
st.markdown(f"""
<div style="background-color:#f0f2f6; padding:15px; border-radius:10px; margin-bottom:20px">
<b>实验环境：</b> 节点数 {n_val} | 传播率 {t_val} | 监测率 {det_val} <br>
<b>图例说明：</b> 🌟 金星:真源头 | ✖️ 紫叉:预测源头 | 🔴 红色:真实感染者 | ⚪ 灰色:已隔离区 | 🟠 橙色:高风险(P > {alpha_val})
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])

# --- 6. 核心算法逻辑 ---
def iterate_step():
    s = st.session_state
    if s.is_contained: return

    # A. 传播：漏网者传播
    leaked_spreaders = list(s.infected - s.quarantined)
    if len(leaked_spreaders) == 0 and s.step > 0:
        s.is_contained = True
        return

    new_inf = set()
    for u in leaked_spreaders:
        for v in s.g.neighbors(u):
            if v not in s.infected and v not in s.quarantined:
                if np.random.rand() < t_val: new_inf.add(v)
    s.infected.update(new_inf)

    # B. 实时抓捕 (每一帧重算)
    if s.step >= hunt_val:
        newly_detected = {n for n in (s.infected - s.quarantined) if np.random.rand() < det_val}
        s.known_cases.update(newly_found := newly_detected)
        
        if len(s.known_cases) > 1:
            try:
                # 基于 Steiner Tree 的路径回溯
                st_tree = nx.algorithms.approximation.steiner_tree(s.g, list(s.known_cases), weight='weight')
                s.quarantined.update(st_tree.nodes())
                # 寻找预测源头
                sub = st_tree if nx.is_connected(st_tree) else st_tree.subgraph(max(nx.connected_components(st_tree), key=len))
                s.pred_source = min(nx.eccentricity(sub), key=lambda x: nx.eccentricity(sub)[x])
            except: pass
    s.step += 1

# --- 7. 绘图函数 ---
def draw_map():
    s = st.session_state
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 分类节点
    tp_nodes = s.infected & s.quarantined
    fp_nodes = s.quarantined - s.infected
    fn_nodes = s.infected - s.quarantined
    tn_nodes = set(s.g.nodes()) - s.infected - s.quarantined
    
    high_risk = set()
    for v in tn_nodes:
        leaked_cnt = len([n for n in s.g.neighbors(v) if n in fn_nodes])
        if (1 - (1 - t_val)**leaked_cnt) > alpha_val: high_risk.add(v)

    # 绘图
    nx.draw_networkx_edges(s.g, s.pos, ax=ax, alpha=0.02, edge_color='gray')
    nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(tn_nodes - high_risk), node_color='#87CEEB', node_size=30, ax=ax)
    nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(fn_nodes), node_color='red', node_size=80, edgecolors='black', ax=ax)
    if tp_nodes:
        nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(tp_nodes), node_color='#A9A9A9', node_size=120, ax=ax)
        nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(tp_nodes), node_color='red', node_size=25, ax=ax)
    if fp_nodes: 
        nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(fp_nodes), node_color='#D3D3D3', node_size=60, ax=ax)
    if high_risk: 
        nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(high_risk), node_color='orange', node_size=80, ax=ax)
    
    # 标注源头 (英文标签避开乱码)
    ax.scatter(s.pos[s.true_source][0], s.pos[s.true_source][1], marker='*', s=400, c='gold', edgecolors='black', zorder=10, label="True Source")
    if s.pred_source:
        ax.scatter(s.pos[s.pred_source][0], s.pos[s.pred_source][1], marker='x', s=300, c='purple', zorder=11, label="Predicted Source")
    
    err = nx.shortest_path_length(s.g, s.true_source, s.pred_source) if s.pred_source else "N/A"
    ax.set_title(f"Step: {s.step} | Error Distance: {err}", fontsize=14)
    ax.axis('off')
    return fig, len(tp_nodes), len(fp_nodes), len(fn_nodes), len(tn_nodes)

# --- 8. 执行渲染循环 ---
map_placeholder = col1.empty()
table_placeholder = col2.empty()
metrics_placeholder = col2.empty()

if st.sidebar.button("▶️ 开始/继续 动态模拟"):
    st.session_state.running = True

if st.session_state.running:
    while not st.session_state.is_contained:
        iterate_step()
        fig, tp, fp, fn, tn = draw_map()
        map_placeholder.pyplot(fig)
        plt.close(fig)

        with table_placeholder.container():
            st.write("#### 📊 2x2 混淆矩阵")
            data = {
                "In-Quarantine (Q)": [f"TP: {tp}", f"FP: {fp}"],
                "Non-Quarantine": [f"FN: {fn}", f"TN: {tn}"]
            }
            df = pd.DataFrame(data, index=["Sick (Red)", "Safe (Blue)"])
            st.table(df)
        
        with metrics_placeholder.container():
            acc = (tp + tn) / n_val
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            st.metric("实时准确率 (Accuracy)", f"{acc:.1%}")
            st.metric("查全率 (Recall/Power)", f"{rec:.1%}")
        
        if st.session_state.is_contained:
            st.success("🎉 疫情已完全控制！所有传染源已被隔离。")
            st.balloons()
            st.session_state.running = False
            break
        time.sleep(0.05)
else:
    # 初始显示
    fig, tp, fp, fn, tn = draw_map()
    map_placeholder.pyplot(fig)
    plt.close(fig)
    st.info("请点击左侧或右侧的开始按钮启动动态模拟过程。")