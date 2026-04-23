import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans

# --- 1. 环境配置 (彻底解决乱码：图表内部使用英文，UI使用中文) ---
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False 

# --- 2. 增强型网络生成 ---
@st.cache_resource
def get_network(n, m, seed):
    g = nx.barabasi_albert_graph(n, m, seed=seed)
    # 增加 k 值让节点分布更稀疏，增加 iterations 让布局更稳定
    pos = nx.spring_layout(g, k=0.25, iterations=50, seed=seed)
    communities = nx.community.louvain_communities(g, seed=seed)
    partition = {node: i for i, comm in enumerate(communities) for node in comm}
    return g, pos, partition

# --- 3. 核心算法：肘部法则 ---
def analyze_elbow(coords, max_k=8):
    if len(coords) < 4: return 1, [0]
    k_range = range(1, min(len(coords), max_k + 1))
    sse = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(coords)
        sse.append(km.inertia_)
    
    # 寻找斜率变化最大的点
    if len(sse) >= 3:
        p1, p2 = np.array([1, sse[0]]), np.array([len(sse), sse[-1]])
        dists = [np.abs(np.cross(p2-p1, p1-np.array([i+1, sse[i]]))) / np.linalg.norm(p2-p1) for i in range(len(sse))]
        best_k = k_range[np.argmax(dists)]
    else:
        best_k = 1
    return best_k, sse

# --- 4. 侧边栏设置 ---
st.sidebar.header("⚙️ 实验参数设置")
seed_val = st.sidebar.number_input("随机种子 (Seed)", value=42)
n_val = st.sidebar.slider("节点数量 (N)", 300, 1500, 600)
m_val = st.sidebar.slider("网络连接密度 (M)", 1, 3, 2)
source_count = st.sidebar.slider("真实源头数量", 1, 5, 3)
t_val = st.sidebar.slider("传播概率 (T)", 0.05, 0.4, 0.15)
det_val = st.sidebar.slider("流调检测率 (Recall)", 0.1, 1.0, 0.6)

if 'step' not in st.session_state or st.sidebar.button("🔄 重置模拟"):
    g, pos, partition = get_network(n_val, m_val, seed_val)
    np.random.seed(seed_val)
    true_sources = list(np.random.choice(list(g.nodes()), source_count, replace=False))
    st.session_state.update({
        'step': 0, 'running': False, 'g': g, 'pos': pos, 'partition': partition,
        'true_sources': true_sources, 'infected': set(true_sources),
        'quarantined': set(), 'known_cases': set(), 'pred_sources': [],
        'sse': [], 'estimated_k': 1, 'is_contained': False
    })

s = st.session_state

# --- 5. 仿真逻辑 ---
def iterate_step():
    if s.is_contained: return
    # 传播
    leaked = list(s.infected - s.quarantined)
    if not leaked and s.step > 0: s.is_contained = True; return
    for u in leaked:
        for v in s.g.neighbors(u):
            if v not in s.infected and v not in s.quarantined and np.random.rand() < t_val:
                s.infected.add(v)
    # 检测与溯源 (多源头逻辑)
    newly_found = {n for n in (s.infected - s.quarantined) if np.random.rand() < det_val}
    s.known_cases.update(newly_found)
    
    if len(s.known_cases) >= 3:
        coords = np.array([s.pos[node] for node in s.known_cases])
        s.estimated_k, s.sse = analyze_elbow(coords)
        km = KMeans(n_clusters=s.estimated_k, n_init=10).fit(coords)
        s.pred_sources = []
        for i in range(s.estimated_k):
            cluster = [list(s.known_cases)[j] for j, label in enumerate(km.labels_) if label == i]
            if cluster:
                try:
                    st_tree = nx.algorithms.approximation.steiner_tree(s.g, cluster)
                    s.quarantined.update(st_tree.nodes())
                    sub = st_tree if nx.is_connected(st_tree) else st_tree.subgraph(max(nx.connected_components(st_tree), key=len))
                    s.pred_sources.append(min(nx.eccentricity(sub), key=nx.eccentricity(sub).get))
                except: pass
    s.step += 1

# --- 6. 绘图函数优化 ---
def draw_main_map():
    fig, ax = plt.subplots(figsize=(10, 8))
    # 1. 绘制极淡的边
    nx.draw_networkx_edges(s.g, s.pos, ax=ax, alpha=0.02, edge_color='gray')
    # 2. 绘制健康节点 (分区域颜色)
    clean = set(s.g.nodes()) - s.infected - s.quarantined
    nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(clean), 
                           node_color=[s.partition[n] for n in clean],
                           cmap=plt.cm.Pastel1, node_size=20, ax=ax, alpha=0.8)
    # 3. 绘制隔离区 (增强视觉感)
    if s.quarantined:
        nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(s.quarantined), 
                               node_color='lightgray', node_size=100, alpha=0.3, ax=ax)
    # 4. 绘制活跃感染者
    nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(s.infected - s.quarantined), 
                           node_color='red', node_size=40, edgecolors='white', ax=ax)
    # 5. 标注真实源头 (金星)
    for ts in s.true_sources:
        ax.scatter(s.pos[ts][0], s.pos[ts][1], marker='*', s=450, c='gold', edgecolors='black', zorder=10)
    # 6. 标注预测源头 (大紫叉)
    for ps in s.pred_sources:
        ax.scatter(s.pos[ps][0], s.pos[ps][1], marker='x', s=250, c='purple', linewidths=3, zorder=11)
    
    ax.set_title(f"Epidemic Spread Step: {s.step} (Red=Infected, Gray=Quarantined)", fontsize=14)
    ax.axis('off')
    return fig

def draw_elbow_plot():
    fig, ax = plt.subplots(figsize=(5, 3.5))
    if len(s.sse) > 1:
        ax.plot(range(1, len(s.sse)+1), s.sse, 'bo-', markersize=4)
        ax.axvline(x=s.estimated_k, color='red', linestyle='--', label=f'Best K={s.estimated_k}')
        ax.set_title("Elbow Method (Source Detection)", fontsize=10)
        ax.set_xlabel("Number of Sources (K)")
        ax.set_ylabel("SSE")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Collecting Data...", ha='center')
    plt.tight_layout()
    return fig

# --- 7. 主界面布局 ---
st.title("🛡️ 多源头图传播监测与精准防控系统")

c1, c2 = st.columns([2, 1])

if st.sidebar.button("▶️ 开启动态模拟"):
    s.running = True

map_ph = c1.empty()
elbow_ph = c2.empty()
matrix_ph = c2.empty()

if s.running:
    while not s.is_contained:
        iterate_step()
        map_ph.pyplot(draw_main_map())
        elbow_ph.pyplot(draw_elbow_plot())
        
        with matrix_ph.container():
            st.markdown("### 📊 隔离效果混淆矩阵")
            tp = len(s.infected & s.quarantined)
            fp = len(s.quarantined - s.infected)
            fn = len(s.infected - s.quarantined)
            tn = n_val - len(s.infected | s.quarantined)
            
            df = pd.DataFrame({
                "被隔离 (Q)": [f"TP: {tp}", f"FP: {fp}"],
                "未隔离 (Free)": [f"FN: {fn}", f"TN: {tn}"]
            }, index=["真实感染 (Sick)", "健康人口 (Safe)"])
            st.table(df)
            st.metric("捕捉率 (Recall)", f"{tp/(tp+fn+1e-5):.1%}")
        
        if s.step > 100: break
        time.sleep(0.05)
else:
    map_ph.pyplot(draw_main_map())
    elbow_ph.pyplot(draw_elbow_plot())
