import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans

# --- 0. 基础设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False 

# --- 1. 核心算法：带 SSE 返回的肘部算法 ---
def analyze_elbow(coords, max_k=8):
    if len(coords) < 3:
        return 1, [0], 0
    
    k_range = range(1, min(len(coords), max_k + 1))
    sse = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(coords)
        sse.append(km.inertia_)
    
    if len(sse) < 3:
        return 1, sse, 0

    # 几何法寻找拐点 (点到斜线距离最远点)
    p1 = np.array([k_range[0], sse[0]])
    p2 = np.array([k_range[-1], sse[-1]])
    dists = []
    for i in range(len(sse)):
        p0 = np.array([k_range[i], sse[i]])
        d = np.abs(np.cross(p2-p1, p1-p0)) / np.linalg.norm(p2-p1)
        dists.append(d)
    
    best_k = k_range[np.argmax(dists)]
    return best_k, sse, best_k

# --- 2. 缓存网络生成 ---
@st.cache_resource
def get_network(n, m, seed):
    g = nx.barabasi_albert_graph(n, m, seed=seed)
    pos = nx.spring_layout(g, k=0.15, iterations=30, seed=seed)
    # 预设社区
    communities = nx.community.louvain_communities(g, seed=seed)
    partition = {node: i for i, comm in enumerate(communities) for node in comm}
    return g, pos, partition

# --- 3. 侧边栏 ---
st.sidebar.header("⚙️ 实验参数设置")
seed_val = st.sidebar.number_input("随机种子 (Seed)", value=42, step=1)
n_val = st.sidebar.slider("节点数量", 300, 1200, 600)
m_val = st.sidebar.slider("网络稠密度", 1, 3, 2)
real_source_count = st.sidebar.slider("真实源头数", 1, 5, 3)
t_val = st.sidebar.slider("传播概率 (T)", 0.05, 0.4, 0.15)
det_val = st.sidebar.slider("流调检测率 (Recall)", 0.1, 1.0, 0.6)
hunt_val = st.sidebar.slider("开始溯源步数", 1, 10, 3)

# --- 4. 初始化 Session State ---
if 'step' not in st.session_state or st.sidebar.button("🔄 重置并运行"):
    g, pos, partition = get_network(n_val, m_val, seed_val)
    np.random.seed(seed_val)
    true_sources = list(np.random.choice(list(g.nodes()), real_source_count, replace=False))
    
    st.session_state.update({
        'step': 0,
        'running': False,
        'g': g,
        'pos': pos,
        'partition': partition,
        'true_sources': true_sources,
        'infected': set(true_sources),
        'quarantined': set(),
        'known_cases': set(),
        'pred_sources': [],
        'sse_history': [],
        'estimated_k': 1,
        'is_contained': False
    })

s = st.session_state

# --- 5. 仿真逻辑 ---
def iterate_step():
    if s.is_contained: return
    # 传播
    spreaders = list(s.infected - s.quarantined)
    if not spreaders and s.step > 0: s.is_contained = True; return
    
    new_inf = {v for u in spreaders for v in s.g.neighbors(u) 
               if v not in s.infected and v not in s.quarantined and np.random.rand() < t_val}
    s.infected.update(new_inf)

    # 溯源与隔离
    if s.step >= hunt_val:
        newly_found = {n for n in (s.infected - s.quarantined) if np.random.rand() < det_val}
        s.known_cases.update(newly_found)
        
        if len(s.known_cases) >= 3:
            case_list = list(s.known_cases)
            coords = np.array([s.pos[node] for node in case_list])
            
            # 肘部算法分析
            best_k, sse, _ = analyze_elbow(coords)
            s.estimated_k = best_k
            s.sse_history = sse
            
            # K-Means 聚类溯源
            km = KMeans(n_clusters=best_k, n_init=10).fit(coords)
            s.pred_sources = []
            for i in range(best_k):
                cluster = [case_list[j] for j, label in enumerate(km.labels_) if label == i]
                if cluster:
                    try:
                        st_tree = nx.algorithms.approximation.steiner_tree(s.g, cluster)
                        s.quarantined.update(st_tree.nodes())
                        # 取中心点
                        sub = st_tree if nx.is_connected(st_tree) else st_tree.subgraph(max(nx.connected_components(st_tree), key=len))
                        ecc = nx.eccentricity(sub)
                        s.pred_sources.append(min(ecc, key=ecc.get))
                    except: pass
    s.step += 1

# --- 6. 绘图函数 ---
def draw_plots():
    # A. 传播主图
    fig_main, ax = plt.subplots(figsize=(8, 6))
    clean = set(s.g.nodes()) - s.infected - s.quarantined
    nx.draw_networkx_edges(s.g, s.pos, ax=ax, alpha=0.01, edge_color='gray')
    nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(clean), node_color=[s.partition[n] for n in clean],
                           cmap=plt.cm.Pastel2, node_size=30, ax=ax)
    if s.quarantined:
        nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(s.quarantined), node_color='gray', alpha=0.3, node_size=80, ax=ax)
    nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(s.infected - s.quarantined), node_color='red', node_size=40, ax=ax)
    for ts in s.true_sources:
        ax.scatter(s.pos[ts][0], s.pos[ts][1], marker='*', s=250, c='gold', edgecolors='black', zorder=5)
    for ps in s.pred_sources:
        ax.scatter(s.pos[ps][0], s.pos[ps][1], marker='x', s=150, c='purple', linewidths=2, zorder=6)
    ax.set_title(f"第 {s.step} 步传播实况", fontsize=12)
    ax.axis('off')

    # B. 肘部算法曲线图
    fig_elbow, ax_e = plt.subplots(figsize=(5, 3))
    if len(s.sse_history) > 1:
        ax_e.plot(range(1, len(s.sse_history)+1), s.sse_history, 'bo-')
        ax_e.axvline(x=s.estimated_k, color='r', linestyle='--', label=f'肘部 K={s.estimated_k}')
        ax_e.set_xlabel("源头数量 K")
        ax_e.set_ylabel("SSE (簇内误差和)")
        ax_e.set_title("肘部算法动态分析")
        ax_e.legend()
    else:
        ax_e.text(0.5, 0.5, "等待流调数据...", ha='center')
    
    return fig_main, fig_elbow

# --- 7. 主界面布局 ---
st.title("🛡️ 多源头图传播监测、肘部算法溯源与混淆矩阵分析")

col_main, col_stats = st.columns([2, 1])

if st.sidebar.button("▶️ 开始模拟"):
    s.running = True

map_ph = col_main.empty()
elbow_ph = col_stats.empty()
matrix_ph = col_stats.empty()

if s.running:
    while not s.is_contained:
        iterate_step()
        f_main, f_elbow = draw_plots()
        map_ph.pyplot(f_main)
        elbow_ph.pyplot(f_elbow)
        plt.close(f_main)
        plt.close(f_elbow)

        # C. 混淆矩阵计算
        with matrix_ph.container():
            st.write("#### 📊 2x2 混淆矩阵 (隔离准确度)")
            tp = len(s.infected & s.quarantined)
            fp = len(s.quarantined - s.infected)
            fn = len(s.infected - s.quarantined)
            tn = n_val - len(s.infected | s.quarantined)
            
            conf_df = pd.DataFrame({
                "被隔离 (Quarantined)": [f"TP: {tp}", f"FP: {fp}"],
                "未隔离 (Free)": [f"FN: {fn}", f"TN: {tn}"]
            }, index=["真实感染 (Sick)", "健康人口 (Safe)"])
            st.table(conf_df)
            
            acc = (tp + tn) / n_val
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            st.write(f"**实时准确率 (Acc):** {acc:.1%}")
            st.write(f"**捕捉率 (Recall):** {rec:.1%}")

        if s.step > 80: break
        time.sleep(0.05)
else:
    f_main, f_elbow = draw_plots()
    map_ph.pyplot(f_main)
    elbow_ph.pyplot(f_elbow)
