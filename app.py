import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde

# --- 1. 页面配置 ---
st.set_page_config(page_title="多源分区热力监控系统", layout="wide")

# --- 2. 核心：强制地理分区布局 (防止聚集成团) ---
@st.cache_resource
def get_isolated_islands(n, m, k_centers, seed):
    np.random.seed(seed)
    g = nx.connected_watts_strogatz_graph(n, k=m*2, p=0.1, seed=seed)
    
    # 强制将中心点拉开极大距离
    theta = np.linspace(0, 2*np.pi, k_centers, endpoint=False)
    # 增加半径到 5.0，确保物理隔离
    centers = np.column_stack([np.cos(theta), np.sin(theta)]) * 5.0 
    
    pos = {}
    nodes = list(g.nodes())
    node_to_island = {}
    
    for i, node in enumerate(nodes):
        c_idx = i % k_centers
        node_to_island[node] = c_idx
        # 在各自的岛屿中心附近随机分布
        pos[node] = centers[c_idx] + np.random.normal(0, 0.8, 2)
        
    return g, pos, node_to_island

# --- 3. 参数设置 ---
st.sidebar.header("⚙️ 模拟参数")
n_val = st.sidebar.slider("总人口 (N)", 500, 2000, 1000)
m_val = st.sidebar.slider("连接密度 (M)", 1, 5, 2)
t_val = st.sidebar.slider("传播概率 (T)", 0.05, 0.5, 0.15)
x_rounds = st.sidebar.slider("干预启动轮次 (X)", 1, 20, 6)
src_count = st.sidebar.slider("源头数量", 1, 8, 3)
det_val = st.sidebar.slider("监测率 (Recall)", 0.1, 1.0, 0.7)
seed_val = st.sidebar.number_input("随机种子", value=42)

# 初始化/重置
params_key = f"{n_val}-{m_val}-{seed_val}-{src_count}"
if 'last_key' not in st.session_state or st.session_state.last_key != params_key:
    st.session_state.last_key = params_key
    g, pos, island_map = get_isolated_islands(n_val, m_val, src_count, seed_val)
    np.random.seed(seed_val)
    # 每个岛屿选一个源头
    true_sources = []
    for i in range(src_count):
        island_nodes = [n for n, isl in island_map.items() if isl == i]
        true_sources.append(np.random.choice(island_nodes))
    
    st.session_state.update({
        'step': 0, 'running': False, 'g': g, 'pos': pos, 'island_map': island_map,
        'true_sources': true_sources,
        'infected': set(true_sources),
        'quarantined': set(),
        'known_cases': set(),
        'node_src_map': {n: island_map[n] for n in true_sources},
        'pred_sources': [], 'avg_err': 0.0, 'sse': [], 'est_k': 1, 'is_contained': False
    })

s = st.session_state

# --- 4. 演化逻辑 ---
def iterate_step():
    if s.is_contained: return
    
    # A. 同步传播
    active = [n for n in s.infected if n not in s.quarantined]
    if not active and s.step > 0: s.is_contained = True; return

    new_inf = {}
    for u in active:
        u_src = s.node_src_map.get(u, 0)
        for v in s.g.neighbors(u):
            if v not in s.infected and v not in s.quarantined:
                if np.random.rand() < t_val: new_inf[v] = u_src
    
    for v, sid in new_inf.items():
        s.infected.add(v)
        s.node_src_map[v] = sid

    # B. 隔离干预 (Step >= X)
    if s.step >= x_rounds:
        detected = {n for n in (s.infected - s.quarantined) if np.random.rand() < det_val}
        s.known_cases.update(detected)
        
        if len(s.known_cases) >= 3:
            coords = np.array([s.pos[n] for n in s.known_cases])
            k_range = range(1, min(len(s.known_cases), 10))
            sse = [KMeans(n_clusters=k, n_init=3).fit(coords).inertia_ for k in k_range]
            s.sse = sse
            
            if len(sse) >= 3:
                # 肘部算法计算
                dists = [np.abs(np.cross(np.array([len(sse),sse[-1]])-np.array([1,sse[0]]), np.array([1,sse[0]])-np.array([i+1,s]))) for i,s in enumerate(sse)]
                s.est_k = k_range[np.argmax(dists)]
                
                km = KMeans(n_clusters=s.est_k, n_init=5).fit(coords)
                s.pred_sources = []
                for i in range(s.est_k):
                    nodes = [list(s.known_cases)[j] for j, l in enumerate(km.labels_) if l == i]
                    if nodes:
                        try:
                            st_tree = nx.algorithms.approximation.steiner_tree(s.g, nodes)
                            s.quarantined.update(st_tree.nodes())
                            sub = st_tree if nx.is_connected(st_tree) else st_tree.subgraph(max(nx.connected_components(st_tree), key=len))
                            s.pred_sources.append(min(nx.eccentricity(sub), key=nx.eccentricity(sub).get))
                        except: pass
            
            if s.pred_sources:
                errs = [min([nx.shortest_path_length(s.g, ts, ps) for ps in s.pred_sources]) for ts in s.true_sources]
                s.avg_err = np.mean(errs)
    s.step += 1

# --- 5. 渲染引擎 ---
def draw_map():
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#ffffff')
    
    # 限制显示范围，防止岛屿太远看不见
    ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)

    # A. KDE 热力图层
    inf_nodes = list(s.infected - s.quarantined)
    if len(inf_nodes) >= 5:
        try:
            inf_coords = np.array([s.pos[n] for n in inf_nodes])
            xx, yy = np.mgrid[-8:8:100j, -8:8:100j]
            kernel = gaussian_kde(inf_coords.T)
            f = np.reshape(kernel(np.vstack([xx.ravel(), yy.ravel()])).T, xx.shape)
            ax.contourf(xx, yy, f, cmap='Spectral_r', alpha=0.4, levels=15, zorder=0)
            ax.contour(xx, yy, f, colors='black', alpha=0.1, levels=10, linewidths=0.5, zorder=1)
        except: pass

    # B. 节点渲染
    # 1. 健康人 (天蓝)
    h_free = [n for n in s.g.nodes() if n not in s.infected and n not in s.quarantined]
    if h_free:
        h_pos = np.array([s.pos[n] for n in h_free])
        ax.scatter(h_pos[:,0], h_pos[:,1], c='#87CEEB', s=20, alpha=0.5, zorder=2)

    # 2. 隔离光晕 (深灰环)
    if s.quarantined:
        q_pos = np.array([s.pos[n] for n in s.quarantined])
        ax.scatter(q_pos[:,0], q_pos[:,1], facecolors='none', edgecolors='#444444', s=120, alpha=0.2, linewidths=2, zorder=3)

    # 3. 感染者核心 (按源头染色)
    if s.infected:
        cmap = plt.cm.get_cmap('tab10', 10)
        for n in s.infected:
            color = cmap(s.node_src_map.get(n, 0) % 10)
            ax.scatter(s.pos[n][0], s.pos[n][1], c=[color], s=35, 
                       edgecolors='black' if n in s.quarantined else 'white', 
                       linewidths=1 if n in s.quarantined else 0.5, zorder=4)

    # 4. 源头标注
    for ts in s.true_sources:
        ax.scatter(s.pos[ts][0], s.pos[ts][1], marker='*', s=400, c='gold', edgecolors='black', zorder=100)
    for ps in s.pred_sources:
        ax.scatter(s.pos[ps][0], s.pos[ps][1], marker='x', s=250, c='purple', linewidths=3, zorder=101)

    ax.set_title(f"Step {s.step} | Multi-Island Epidemic Monitor", fontsize=15)
    ax.axis('off')
    return fig

# --- 6. 界面展示 ---
st.title("🛡️ 多岛屿分区传播与动态溯源仿真")

col_main, col_side = st.columns([3.5, 1.2])
map_spot = col_main.empty()
side_container = col_side.container()

if st.sidebar.button("▶️ 开启演化"):
    s.running = True

if s.running:
    while not s.is_contained and s.running:
        iterate_step()
        map_spot.pyplot(draw_map())
        
        with side_container:
            # 1. 肘部图
            fig_e, ax_e = plt.subplots(figsize=(5, 3.5))
            if len(s.sse) > 1:
                ax_e.plot(range(1, len(s.sse)+1), s.sse, 'ro-', markersize=4)
                ax_e.axvline(x=s.est_k, color='blue', linestyle='--')
                ax_e.set_title("Elbow Method (Source K)")
            st.pyplot(fig_e)
            plt.close(fig_e)

            # 2. 混淆矩阵
            st.markdown("#### 📊 实时数据")
            tp = len(s.infected & s.quarantined)
            fp = len(s.quarantined - s.infected)
            fn = len(s.infected - s.quarantined)
            tn = n_val - len(s.infected | s.quarantined)
            st.table(pd.DataFrame({"Isolated": [tp, fp], "Free": [fn, tn]}, index=["Sick", "Healthy"]))
            
            # 3. 指标
            st.metric("平均溯源距离误差", f"{s.avg_err:.2f} Hops")
            st.metric("捕捉率 (Recall)", f"{tp/(tp+fn+1e-5):.1%}")
            st.write(f"当前步数: {s.step}")
        
        plt.close('all')
        time.sleep(0.01)
else:
    map_spot.pyplot(draw_map())
