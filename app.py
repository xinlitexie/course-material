import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde
from matplotlib.collections import LineCollection

# --- 1. 页面配置 ---
st.set_page_config(page_title="分区热力图传播仿真", layout="wide")

# --- 2. 核心：强制“群岛”分区布局 ---
@st.cache_resource
def get_island_network(n, m, k_centers, seed):
    np.random.seed(seed)
    # 创建图
    g = nx.barabasi_albert_graph(n, m, seed=seed)
    
    # --- 强制分区逻辑 ---
    # 为每个源头设定一个远离的中心点
    theta = np.linspace(0, 2*np.pi, k_centers, endpoint=False)
    centers = np.column_stack([np.cos(theta), np.sin(theta)]) * 1.5 
    
    pos = {}
    nodes = list(g.nodes())
    # 将节点均匀分配到各个中心
    for i, node in enumerate(nodes):
        c_idx = i % k_centers
        # 初始位置在中心附近抖动
        pos[node] = centers[c_idx] + np.random.normal(0, 0.25, 2)
    
    # 使用 spring_layout 微调，但限制迭代次数防止聚拢
    pos = nx.spring_layout(g, pos=pos, k=0.1, iterations=15, fixed=None, seed=seed)
    
    # 彻底检查坐标有效性，防止 NaN
    for node in list(pos.keys()):
        if np.any(np.isnan(pos[node])):
            pos[node] = np.array([0.0, 0.0])
            
    return g, pos

# --- 3. 参数设置 ---
st.sidebar.header("⚙️ 模拟参数")
n_val = st.sidebar.slider("总人口 (N)", 500, 3000, 1500)
m_val = st.sidebar.slider("网络密度 (M)", 1, 3, 2)
t_val = st.sidebar.slider("传播概率 (T)", 0.01, 0.4, 0.15)
x_rounds = st.sidebar.slider("干预启动轮次 (X)", 1, 30, 8)
src_count = st.sidebar.slider("源头数量", 1, 8, 4)
det_val = st.sidebar.slider("监测率", 0.1, 1.0, 0.7)
seed_val = st.sidebar.number_input("随机种子", value=42)

# 初始化与重置
params_key = f"{n_val}-{m_val}-{seed_val}-{src_count}"
if 'last_key' not in st.session_state or st.session_state.last_key != params_key:
    st.session_state.last_key = params_key
    g, pos = get_island_network(n_val, m_val, src_count, seed_val)
    np.random.seed(seed_val)
    # 初始源头：从每个群岛中心选一个
    true_sources = [list(g.nodes())[i] for i in range(src_count)]
    
    st.session_state.update({
        'step': 0, 'running': False, 'g': g, 'pos': pos,
        'true_sources': true_sources,
        'infected': set(true_sources),
        'quarantined': set(),
        'known_cases': set(),
        'node_src_map': {n: i for i, n in enumerate(true_sources)},
        'pred_sources': [], 'avg_err': 0.0, 'is_contained': False
    })

s = st.session_state

# --- 4. 演化引擎 ---
def iterate_step():
    if s.is_contained: return
    
    # A. 传播
    current_active = [n for n in s.infected if n not in s.quarantined]
    if not current_active and s.step > 0: s.is_contained = True; return

    new_inf = {}
    for u in current_active:
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
        
        if len(s.known_cases) >= 5:
            coords = np.array([s.pos[n] for n in s.known_cases])
            k_range = range(1, min(len(s.known_cases), 10))
            sse = [KMeans(n_clusters=k, n_init=3).fit(coords).inertia_ for k in k_range]
            
            # 推断源头数
            if len(sse) >= 3:
                dists = [np.abs(np.cross(np.array([len(sse),sse[-1]])-np.array([1,sse[0]]), np.array([1,sse[0]])-np.array([i+1,s]))) for i,s in enumerate(sse)]
                est_k = k_range[np.argmax(dists)]
                
                km = KMeans(n_clusters=est_k, n_init=5).fit(coords)
                s.pred_sources = []
                for i in range(est_k):
                    nodes = [list(s.known_cases)[j] for j, label in enumerate(km.labels_) if label == i]
                    if nodes:
                        try:
                            st_tree = nx.algorithms.approximation.steiner_tree(s.g, nodes)
                            s.quarantined.update(st_tree.nodes())
                            sub = st_tree if nx.is_connected(st_tree) else st_tree.subgraph(max(nx.connected_components(st_tree), key=len))
                            s.pred_sources.append(min(nx.eccentricity(sub), key=nx.eccentricity(sub).get))
                        except: pass
            
            # 误差计算
            if s.pred_sources:
                errs = []
                for ts in s.true_sources:
                    try:
                        errs.append(min([nx.shortest_path_length(s.g, ts, ps) for ps in s.pred_sources]))
                    except: errs.append(15)
                s.avg_err = np.mean(errs)
    s.step += 1

# --- 5. 渲染引擎：解决崩溃并增强分区 ---
def draw_map():
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_facecolor('#ffffff')
    
    # A. 渲染 KDE 热力等高线 (核心分区视觉)
    active_inf = list(s.infected - s.quarantined)
    if len(active_inf) >= 5:
        try:
            inf_coords = np.array([s.pos[n] for n in active_inf])
            xmin, xmax = -2.5, 2.5
            ymin, ymax = -2.5, 2.5
            xx, yy = np.mgrid[xmin:xmax:80j, ymin:ymax:80j]
            kernel = gaussian_kde(inf_coords.T)
            f = np.reshape(kernel(np.vstack([xx.ravel(), yy.ravel()])).T, xx.shape)
            
            # 仿地图填充
            ax.contourf(xx, yy, f, cmap='Spectral_r', alpha=0.4, levels=15, zorder=0)
            # 强化等高线线条
            ax.contour(xx, yy, f, colors='black', alpha=0.15, levels=10, linewidths=0.6, zorder=1)
        except: pass

    # B. 安全绘制连边 (不使用 nx.draw_networkx_edges)
    # 仅绘制 10% 的边以防崩溃
    try:
        edges = list(s.g.edges())
        if len(edges) > 0:
            edge_coords = [[s.pos[u], s.pos[v]] for u, v in edges[:int(len(edges)*0.1)]]
            lc = LineCollection(edge_coords, colors='#DDDDDD', linewidths=0.2, alpha=0.1, zorder=2)
            ax.add_collection(lc)
    except: pass
    
    # C. 节点渲染
    nodes_all = list(s.g.nodes())
    pos_arr = np.array([s.pos[n] for n in nodes_all])
    
    # 健康人
    h_free = [n for n in nodes_all if n not in s.infected and n not in s.quarantined]
    if h_free:
        h_pos = np.array([s.pos[n] for n in h_free])
        ax.scatter(h_pos[:,0], h_pos[:,1], c='#87CEEB', s=15, alpha=0.4, edgecolors='none', zorder=3)

    # 隔离光晕 (灰色)
    if s.quarantined:
        q_pos = np.array([s.pos[n] for n in s.quarantined])
        ax.scatter(q_pos[:,0], q_pos[:,1], c='#333333', s=100, alpha=0.1, edgecolors='none', zorder=4)

    # 感染者核心着色
    cmap = plt.cm.get_cmap('tab10', 10)
    for n in (s.infected):
        color = cmap(s.node_src_map.get(n, 0) % 10)
        is_q = n in s.quarantined
        ax.scatter(s.pos[n][0], s.pos[n][1], c=[color], s=40 if is_q else 25, 
                   edgecolors='black' if is_q else 'none', linewidths=0.5, zorder=5)

    # D. 标注源头
    for ts in s.true_sources:
        ax.scatter(s.pos[ts][0], s.pos[ts][1], marker='*', s=500, c='gold', edgecolors='black', zorder=100)
    for ps in s.pred_sources:
        ax.scatter(s.pos[ps][0], s.pos[ps][1], marker='x', s=250, c='purple', linewidths=3, zorder=101)

    ax.set_title(f"Epidemic Island Regions | Step {s.step} | Error: {s.avg_err:.2f} hops", fontsize=15)
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
    ax.axis('off')
    return fig

# --- 6. UI 主界面 ---
st.title("🛡️ 社交网络分区传播监控 (群岛布局版)")
st.caption("视觉说明：不同源头被分配到独立的地理岛屿。热力图红色代表感染核心，等高线代表扩散层级。")

c1, c2 = st.columns([4, 1.2])
map_spot = c1.empty()
metric_spot = c2.empty()

if st.sidebar.button("▶️ 开启实时演化模拟"):
    s.running = True

if s.running:
    while not s.is_contained and s.running:
        iterate_step()
        f = draw_map()
        map_spot.pyplot(f)
        plt.close(f) # 释放内存
        
        with metric_spot.container():
            st.metric("平均溯源距离误差", f"{s.avg_err:.2f} Hops")
            tp = len(s.infected & s.quarantined)
            fn = len(s.infected - s.quarantined)
            st.metric("捕捉率 (Recall)", f"{tp/(tp+fn+1e-5):.1%}")
            st.write(f"**当前步数:** {s.step}")
        
        time.sleep(0.01)
else:
    map_spot.pyplot(draw_map())
