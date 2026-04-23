import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans

# --- 1. 页面基本配置 ---
st.set_page_config(page_title="多源图传播实时动态监测", layout="wide")

# 强制使用矢量字体避免乱码，图表内部采用英文
plt.rcParams['font.sans-serif'] = ['Arial'] 
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 核心网络生成 (带缓存) ---
@st.cache_resource
def create_static_network(n, m, seed):
    g = nx.barabasi_albert_graph(n, m, seed=seed)
    pos = nx.spring_layout(g, k=0.3, iterations=50, seed=seed)
    communities = nx.community.louvain_communities(g, seed=seed)
    partition = {node: i for i, comm in enumerate(communities) for node in comm}
    return g, pos, partition

# --- 3. 侧边栏参数 ---
st.sidebar.header("🛡️ 系统参数控制面板")
n_val = st.sidebar.slider("总人口 (N)", 200, 1500, 800)
m_val = st.sidebar.slider("连接密度 (M)", 1, 5, 2)
t_val = st.sidebar.slider("传播概率 (T)", 0.05, 0.5, 0.15)
det_val = st.sidebar.slider("流调监测率 (Recall)", 0.1, 1.0, 0.7)
src_count = st.sidebar.slider("真实源头数量", 1, 5, 3)
seed_val = st.sidebar.number_input("随机种子 (Seed)", value=42)

# 关键：当滑块参数改变时，自动触发重置逻辑
params_hash = f"{n_val}-{m_val}-{seed_val}-{src_count}-{t_val}-{det_val}"
if 'last_params' not in st.session_state or st.session_state.last_params != params_hash:
    st.session_state.last_params = params_hash
    g, pos, partition = create_static_network(n_val, m_val, seed_val)
    np.random.seed(seed_val)
    true_sources = list(np.random.choice(list(g.nodes()), src_count, replace=False))
    
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
        'sse': [],
        'est_k': 1,
        'is_contained': False
    })

s = st.session_state

# --- 4. 肘部算法逻辑 ---
def get_optimal_k(coords):
    if len(coords) < 4: return 1, [0]
    ks = range(1, min(len(coords), 9))
    sse = [KMeans(n_clusters=k, n_init=5, random_state=42).fit(coords).inertia_ for k in ks]
    if len(sse) < 3: return 1, sse
    p1, p2 = np.array([1, sse[0]]), np.array([len(sse), sse[-1]])
    dists = [np.abs(np.cross(p2-p1, p1-np.array([i+1, sse[i]]))) / np.linalg.norm(p2-p1) for i in range(len(sse))]
    return ks[np.argmax(dists)], sse

# --- 5. 单步演化逻辑 ---
def run_one_step():
    if s.is_contained: return

    # A. 传播：基于 T 概率向邻居扩散
    current_infected = list(s.infected - s.quarantined)
    if not current_infected and s.step > 0:
        s.is_contained = True
        return

    new_infections = set()
    for u in current_infected:
        for v in s.g.neighbors(u):
            if v not in s.infected and v not in s.quarantined:
                if np.random.rand() < t_val:
                    new_infections.add(v)
    s.infected.update(new_infections)

    # B. 溯源与隔离：基于 Recall 发现病例
    newly_detected = {n for n in (s.infected - s.quarantined) if np.random.rand() < det_val}
    s.known_cases.update(newly_detected)

    if len(s.known_cases) >= 3:
        coords = np.array([s.pos[node] for node in s.known_cases])
        s.est_k, s.sse = get_optimal_k(coords)
        
        km = KMeans(n_clusters=s.est_k, n_init=10).fit(coords)
        s.pred_sources = []
        for i in range(s.est_k):
            cluster = [list(s.known_cases)[j] for j, label in enumerate(km.labels_) if label == i]
            if cluster:
                try:
                    # 使用 Steiner Tree 模拟封闭管理路径
                    st_tree = nx.algorithms.approximation.steiner_tree(s.g, cluster)
                    s.quarantined.update(st_tree.nodes())
                    # 寻找簇中心点
                    sub = st_tree if nx.is_connected(st_tree) else st_tree.subgraph(max(nx.connected_components(st_tree), key=len))
                    ecc = nx.eccentricity(sub)
                    s.pred_sources.append(min(ecc, key=ecc.get))
                except: pass
    s.step += 1

# --- 6. 绘图函数 (解决标记消失与尺寸问题) ---
def render_plots():
    # 主图：演化图
    fig_map, ax = plt.subplots(figsize=(12, 9)) # 显著放大尺寸
    
    # 1. 连边
    nx.draw_networkx_edges(s.g, s.pos, ax=ax, alpha=0.02, edge_color='#CCCCCC')
    
    # 2. 健康节点
    all_nodes = set(s.g.nodes())
    healthy = all_nodes - s.infected - s.quarantined
    nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(healthy), 
                           node_color=[s.partition[n] for n in healthy],
                           cmap=plt.cm.Spectral, node_size=25, alpha=0.5, ax=ax)
    
    # 3. 隔离区 (灰色)
    if s.quarantined:
        nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(s.quarantined), 
                               node_color='#555555', node_size=80, alpha=0.2, ax=ax)

    # 4. 活跃感染者 (红色 - 置于顶层)
    active_inf = s.infected - s.quarantined
    if active_inf:
        nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(active_inf), 
                               node_color='#FF0000', node_size=60, edgecolors='white', linewidths=1, ax=ax)
    
    # 5. 真实源头 (金星)
    for ts in s.true_sources:
        ax.scatter(s.pos[ts][0], s.pos[ts][1], marker='*', s=500, c='#FFD700', edgecolors='black', zorder=100)
    
    # 6. 预测源头 (紫叉)
    for ps in s.pred_sources:
        ax.scatter(s.pos[ps][0], s.pos[ps][1], marker='x', s=300, c='#800080', linewidths=4, zorder=101)

    ax.set_title(f"LIVE STEP: {s.step} | RED = INFECTED | STAR = TRUE SOURCE", fontsize=15)
    ax.axis('off')

    # 肘部图
    fig_elbow, ax_e = plt.subplots(figsize=(5, 4))
    if len(s.sse) > 1:
        ax_e.plot(range(1, len(s.sse)+1), s.sse, 'ro-', linewidth=2)
        ax_e.axvline(x=s.est_k, color='blue', linestyle='--', label=f'Best K={s.est_k}')
        ax_e.set_title("Elbow Method Analysis", fontsize=12)
        ax_e.set_xlabel("Number of Clusters (K)")
        ax_e.legend()
    else:
        ax_e.text(0.5, 0.5, "Waiting for data...", ha='center')
    plt.tight_layout()

    return fig_map, fig_elbow

# --- 7. 主界面布局与动态循环 ---
st.title("🛡️ 多源头图传播实时动态演化系统")

col_left, col_right = st.columns([4, 1.2]) # 调整比例，放大主图展示区

with st.sidebar:
    st.write("---")
    run_btn = st.button("▶️ 启动/继续 实时演化")
    if run_btn:
        s.running = True
    stop_btn = st.button("⏸️ 暂停模拟")
    if stop_btn:
        s.running = False

map_placeholder = col_left.empty()
elbow_placeholder = col_right.empty()
matrix_placeholder = col_right.empty()

# 动态演化循环
if s.running:
    while not s.is_contained and s.running:
        run_one_step()
        f_map, f_elbow = render_plots()
        
        map_placeholder.pyplot(f_map)
        elbow_placeholder.pyplot(f_elbow)
        
        # 混淆矩阵更新
        with matrix_placeholder.container():
            st.markdown("#### 📊 实时混淆矩阵")
            tp = len(s.infected & s.quarantined)
            fp = len(s.quarantined - s.infected)
            fn = len(s.infected - s.quarantined)
            tn = n_val - len(s.infected | s.quarantined)
            
            mx_data = {
                "Quarantined": [f"TP: {tp}", f"FP: {fp}"],
                "Not Quarantined": [f"FN: {fn}", f"TN: {tn}"]
            }
            st.table(pd.DataFrame(mx_data, index=["Sick", "Healthy"]))
            st.metric("Detection Recall", f"{tp/(tp+fn+1e-5):.1%}")
        
        plt.close('all')
        if s.step > 150: s.running = False
        time.sleep(0.05) # 控制刷新频率
else:
    # 静态展示当前状态
    f_map, f_elbow = render_plots()
    map_placeholder.pyplot(f_map)
    elbow_placeholder.pyplot(f_elbow)
    with matrix_placeholder:
        st.info("点击左侧按钮开始演化。调整上方滑块将实时重置实验。")
