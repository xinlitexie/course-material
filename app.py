import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans
try:
    from scipy.stats import gaussian_kde
except ImportError:
    st.error("缺少 scipy 库，请在 requirements.txt 中添加 scipy")

# --- 1. 页面配置 ---
st.set_page_config(page_title="图传播热力等高线监控系统", layout="wide")
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 安全的网络生成 ---
@st.cache_resource
def get_safe_network(n, m, seed):
    # 确保 n > m
    if n <= m: n = m + 1
    g = nx.barabasi_albert_graph(n, m, seed=seed)
    # 增大 k 值让布局更稀疏，模拟地图感
    pos = nx.spring_layout(g, k=0.6/np.sqrt(n/500), iterations=40, seed=seed)
    return g, pos

# --- 3. 侧边栏参数 ---
st.sidebar.header("⚙️ 模拟参数设置")
n_val = st.sidebar.slider("总人口 (N)", 500, 3000, 1000)
m_val = st.sidebar.slider("连接密度 (M)", 1, 5, 2)
t_val = st.sidebar.slider("传播概率 (T)", 0.01, 0.5, 0.12)
x_rounds = st.sidebar.slider("干预启动轮次 (X)", 1, 30, 8)
det_val = st.sidebar.slider("流调监测率 (Recall)", 0.1, 1.0, 0.7)
src_count = st.sidebar.slider("初始源头数量", 1, 8, 3)
seed_val = st.sidebar.number_input("随机种子", value=42)

# 参数自动重置逻辑
params_key = f"{n_val}-{m_val}-{seed_val}-{src_count}"
if 'last_key' not in st.session_state or st.session_state.last_key != params_key:
    st.session_state.last_key = params_key
    g, pos = get_safe_network(n_val, m_val, seed_val)
    np.random.seed(seed_val)
    true_sources = list(np.random.choice(list(g.nodes()), min(src_count, len(g.nodes())), replace=False))
    st.session_state.update({
        'step': 0, 'running': False, 'g': g, 'pos': pos,
        'true_sources': true_sources,
        'infected': set(true_sources),
        'quarantined': set(),
        'known_cases': set(),
        'pred_sources': [],
        'node_source_map': {n: i for i, n in enumerate(true_sources)},
        'sse': [], 'est_k': 0, 'avg_err': 0.0, 'is_contained': False
    })

s = st.session_state

# --- 4. 演化引擎 (严格同步传播) ---
def iterate_step():
    if s.is_contained: return
    
    # A. 同步扩散
    current_active = [n for n in s.infected if n not in s.quarantined]
    if not current_active and s.step > 0:
        s.is_contained = True; return

    newly_infected = {}
    for u in current_active:
        u_src = s.node_source_map.get(u, 0)
        for v in s.g.neighbors(u):
            if v not in s.infected and v not in s.quarantined:
                if np.random.rand() < t_val:
                    newly_infected[v] = u_src
    
    for v, sid in newly_infected.items():
        s.infected.add(v)
        s.node_source_map[v] = sid

    # B. 延迟隔离 (Step >= X)
    if s.step >= x_rounds:
        detected = {n for n in (s.infected - s.quarantined) if np.random.rand() < det_val}
        s.known_cases.update(detected)
        
        if len(s.known_cases) >= 5:
            coords = np.array([s.pos[n] for n in s.known_cases])
            k_range = range(1, min(len(s.known_cases), 10))
            sse_vals = [KMeans(n_clusters=k, n_init=3, random_state=42).fit(coords).inertia_ for k in k_range]
            s.sse = sse_vals
            
            # 肘部算法推断 K
            if len(sse_vals) >= 3:
                p1, p2 = np.array([1, sse_vals[0]]), np.array([len(sse_vals), sse_vals[-1]])
                dists = [np.abs(np.cross(p2-p1, p1-np.array([i+1, sse_vals[i]]))) / np.linalg.norm(p2-p1) for i in range(len(sse_vals))]
                s.est_k = k_range[np.argmax(dists)]
                
                km = KMeans(n_clusters=s.est_k, n_init=5).fit(coords)
                s.pred_sources = []
                for i in range(s.est_k):
                    nodes = [list(s.known_cases)[j] for j, label in enumerate(km.labels_) if label == i]
                    if nodes:
                        try:
                            st_tree = nx.algorithms.approximation.steiner_tree(s.g, nodes)
                            s.quarantined.update(st_tree.nodes())
                            sub = st_tree if nx.is_connected(st_tree) else st_tree.subgraph(max(nx.connected_components(st_tree), key=len))
                            s.pred_sources.append(min(nx.eccentricity(sub), key=nx.eccentricity(sub).get))
                        except: pass
            
            # 计算溯源误差 (跳数)
            if s.pred_sources:
                err_sum = 0
                for ts in s.true_sources:
                    err_sum += min([nx.shortest_path_length(s.g, ts, ps) for ps in s.pred_sources])
                s.avg_err = err_sum / len(s.true_sources)
                
    s.step += 1

# --- 5. 增强渲染逻辑 (热力等高线) ---
def draw_map():
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor('#fdfdfd')
    
    # A. 绘制热力图层 (KDE) - 增加异常保护
    active_inf = list(s.infected - s.quarantined)
    if len(active_inf) >= 4:
        try:
            inf_coords = np.array([s.pos[n] for n in active_inf])
            all_pos_vals = np.array(list(s.pos.values()))
            xmin, xmax = all_pos_vals[:, 0].min()-0.1, all_pos_vals[:, 0].max()+0.1
            ymin, ymax = all_pos_vals[:, 1].min()-0.1, all_pos_vals[:, 1].max()+0.1
            
            xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            kernel = gaussian_kde(inf_coords.T)
            f = np.reshape(kernel(np.vstack([xx.ravel(), yy.ravel()])).T, xx.shape)
            
            # 渲染多级热力场
            ax.contourf(xx, yy, f, cmap='RdYlBu_r', alpha=0.35, levels=12, zorder=0)
            # 渲染明显的等高线线条
            ax.contour(xx, yy, f, colors='gray', alpha=0.15, levels=8, linewidths=0.8, zorder=1)
        except Exception:
            pass # 坐标共线或点数不足时跳过背景渲染

    # B. 绘制网络连边
    nx.draw_networkx_edges(s.g, s.pos, ax=ax, alpha=0.015, edge_color='#CCCCCC', zorder=2)
    
    # C. 绘制节点
    # 1. 健康节点
    h_free = list(set(s.g.nodes()) - s.infected - s.quarantined)
    nx.draw_networkx_nodes(s.g, s.pos, nodelist=h_free, node_color='#ADD8E6', node_size=15, alpha=0.5, ax=ax, zorder=3)

    # 2. 隔离背景 (光晕)
    if s.quarantined:
        nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(s.quarantined), node_color='lightgray', node_size=120, alpha=0.25, ax=ax, zorder=4)

    # 3. 活跃感染者 (按源头 ID 染色)
    inf_free = list(s.infected - s.quarantined)
    inf_q = list(s.infected & s.quarantined)
    cmap_nodes = plt.cm.get_cmap('tab10', max(10, src_count + 1))
    
    for nodes, is_q in [(inf_free, False), (inf_q, True)]:
        if nodes:
            c_list = [cmap_nodes(s.node_source_map.get(n, 0)) for n in nodes]
            nx.draw_networkx_nodes(s.g, s.pos, nodelist=nodes, 
                                   node_color=c_list, node_size=45 if is_q else 30,
                                   edgecolors='black' if is_q else 'white', 
                                   linewidths=1.2 if is_q else 0.5, ax=ax, zorder=5)

    # D. 标注源头 (金星/紫叉)
    for ts in s.true_sources:
        ax.scatter(s.pos[ts][0], s.pos[ts][1], marker='*', s=500, c='gold', edgecolors='black', zorder=100, label="True Source")
    for ps in s.pred_sources:
        ax.scatter(s.pos[ps][0], s.pos[ps][1], marker='x', s=300, c='purple', linewidths=4, zorder=101, label="Est Source")

    ax.set_title(f"Dynamic Heatmap | Step: {s.step} | Avg Err: {s.avg_err:.2f} hops", fontsize=14)
    ax.axis('off')
    return fig

# --- 6. UI 主界面渲染 ---
st.title("🛡️ 多源头图传播热力演化与自动溯源系统")

c1, c2 = st.columns([3.8, 1.2])
map_spot = c1.empty()
elbow_spot = c2.empty()
metric_spot = c2.empty()

if st.sidebar.button("▶️ 开启实时演化模拟"):
    s.running = True

if s.running:
    while not s.is_contained and s.running:
        iterate_step()
        f_map = draw_map()
        map_spot.pyplot(f_map)
        
        # 实时绘制肘部图
        fig_e, ax_e = plt.subplots(figsize=(5, 3.5))
        if len(s.sse) > 1:
            ax_e.plot(range(1, len(s.sse)+1), s.sse, 'ro-', markersize=4)
            ax_e.axvline(x=s.est_k, color='blue', linestyle='--')
            ax_e.set_title("Elbow Method Analysis")
        elbow_spot.pyplot(fig_e)
        
        # 指标更新
        with metric_spot.container():
            st.metric("平均溯源距离误差", f"{s.avg_err:.2f} hops")
            tp = len(s.infected & s.quarantined)
            fn = len(s.infected - s.quarantined)
            st.metric("疫情控制率 (Recall)", f"{tp/(tp+fn+1e-5):.1%}")
            st.write(f"**推断源头数量:** {s.est_k}")
        
        # 内存清理
        plt.close(f_map)
        plt.close(fig_e)
        
        if s.is_contained:
            st.balloons()
            s.running = False
        time.sleep(0.01)
else:
    map_spot.pyplot(draw_map())
