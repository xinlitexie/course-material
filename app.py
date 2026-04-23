import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde

# --- 1. 页面配置 ---
st.set_page_config(page_title="图传播热力等高线可视化", layout="wide")
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 聚类强化布局生成 ---
@st.cache_resource
def get_clustered_network(n, m, src_count, seed):
    g = nx.barabasi_albert_graph(n, m, seed=seed)
    
    # 为了让分区明显，我们给不同的社区一个初始的中心偏移
    pos = nx.spring_layout(g, k=0.5, iterations=50, seed=seed)
    return g, pos

# --- 3. 侧边栏参数 ---
st.sidebar.header("⚙️ 实验参数设置")
n_val = st.sidebar.slider("总人口 (N)", 500, 3000, 1000)
m_val = st.sidebar.slider("连接密度 (M)", 1, 3, 2)
t_val = st.sidebar.slider("传播概率 (T)", 0.01, 0.5, 0.12)
x_rounds = st.sidebar.slider("干预启动轮次 (X)", 1, 30, 6)
det_val = st.sidebar.slider("流调检测率 (Recall)", 0.1, 1.0, 0.7)
src_count = st.sidebar.slider("初始源头数量", 1, 6, 3)
seed_val = st.sidebar.number_input("随机种子", value=42)

# 参数自动重置
params_key = f"{n_val}-{m_val}-{seed_val}-{src_count}"
if 'last_key' not in st.session_state or st.session_state.last_key != params_key:
    st.session_state.last_key = params_key
    g, pos = get_clustered_network(n_val, m_val, src_count, seed_val)
    np.random.seed(seed_val)
    true_sources = list(np.random.choice(list(g.nodes()), src_count, replace=False))
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

# --- 4. 演化引擎 ---
def iterate_step():
    if s.is_contained: return
    
    # A. 同步传播
    spreaders = [n for n in s.infected if n not in s.quarantined]
    if not spreaders and s.step > 0: s.is_contained = True; return

    new_inf = {}
    for u in spreaders:
        u_src = s.node_source_map.get(u, 0)
        for v in s.g.neighbors(u):
            if v not in s.infected and v not in s.quarantined:
                if np.random.rand() < t_val: new_inf[v] = u_src
    for v, sid in new_inf.items():
        s.infected.add(v); s.node_source_map[v] = sid

    # B. 延迟干预
    if s.step >= x_rounds:
        detected = {n for n in (s.infected - s.quarantined) if np.random.rand() < det_val}
        s.known_cases.update(detected)
        if len(s.known_cases) >= 5:
            coords = np.array([s.pos[n] for n in s.known_cases])
            ks = range(1, min(len(s.known_cases), 10))
            sse = [KMeans(n_clusters=k, n_init=5).fit(coords).inertia_ for k in ks]
            s.sse = sse
            if len(sse) >= 3:
                # 肘部法则
                dists = [np.abs(np.cross(np.array([ks[-1],sse[-1]])-np.array([ks[0],sse[0]]), np.array([ks[0],sse[0]])-np.array([k,s]))) for k,s in zip(ks,sse)]
                s.est_k = ks[np.argmax(dists)]
                km = KMeans(n_clusters=s.est_k, n_init=10).fit(coords)
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
            # 计算误差距离
            if s.pred_sources:
                d = 0
                for ts in s.true_sources:
                    d += min([nx.shortest_path_length(s.g, ts, ps) for ps in s.pred_sources])
                s.avg_err = d / len(s.true_sources)
    s.step += 1

# --- 5. 渲染：密度热力图 + 分区显示 ---
def draw_map():
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor('#F0F2F6')

    # 提取坐标
    all_pos = np.array(list(s.pos.values()))
    inf_nodes = list(s.infected)
    
    # --- A. 绘制热力背景 (KDE) ---
    if len(inf_nodes) > 5:
        inf_coords = np.array([s.pos[n] for n in inf_nodes])
        x, y = inf_coords[:, 0], inf_coords[:, 1]
        # 创建网格
        xmin, xmax = all_pos[:, 0].min()-0.1, all_pos[:, 0].max()+0.1
        ymin, ymax = all_pos[:, 1].min()-0.1, all_pos[:, 1].max()+0.1
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        kernel = gaussian_kde(inf_coords.T)
        f = np.reshape(kernel(positions).T, xx.shape)
        # 绘制等高线填充
        ax.contourf(xx, yy, f, cmap='Spectral_r', alpha=0.4, levels=15, zorder=0)
        # 绘制等高线线条 (类似参考图效果)
        ax.contour(xx, yy, f, colors='black', alpha=0.2, levels=10, linewidths=0.5, zorder=1)

    # --- B. 绘制网络层 ---
    nx.draw_networkx_edges(s.g, s.pos, ax=ax, alpha=0.01, edge_color='gray', zorder=2)
    
    # 1. 健康节点
    h_free = set(s.g.nodes()) - s.infected - s.quarantined
    nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(h_free), node_color='#87CEEB', node_size=15, alpha=0.5, ax=ax, zorder=3)

    # 2. 隔离节点 (灰色光晕)
    if s.quarantined:
        nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(s.quarantined), node_color='gray', node_size=100, alpha=0.2, ax=ax, zorder=4)

    # 3. 感染节点 (按源头着色)
    inf_free = s.infected - s.quarantined
    inf_q = s.infected & s.quarantined
    cmap_nodes = plt.cm.get_cmap('tab10', src_count + 1)
    
    for nodes, is_q in [(inf_free, False), (inf_q, True)]:
        if nodes:
            c_list = [cmap_nodes(s.node_source_map.get(n, 0)) for n in nodes]
            nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(nodes), 
                                   node_color=c_list, node_size=40 if is_q else 30,
                                   edgecolors='black' if is_q else 'white', linewidths=0.5, ax=ax, zorder=5)

    # 4. 真实源头 (金星) 与 预测源头 (紫叉)
    for ts in s.true_sources:
        ax.scatter(s.pos[ts][0], s.pos[ts][1], marker='*', s=500, c='gold', edgecolors='black', zorder=100)
    for ps in s.pred_sources:
        ax.scatter(s.pos[ps][0], s.pos[ps][1], marker='x', s=300, c='purple', linewidths=4, zorder=101)

    ax.set_title(f"Epidemic Density Map | Step: {s.step} | Avg Dist: {s.avg_err:.2f} hops", fontsize=15)
    ax.axis('off')
    return fig

# --- 6. UI 主界面 ---
st.title("🛡️ 多源头图传播：密度场演化与分区动态防控系统")

col_main, col_stats = st.columns([4, 1.2])
map_ph = col_main.empty()
elbow_ph = col_stats.empty()
metric_ph = col_stats.empty()

if st.sidebar.button("▶️ 启动/继续 实时演化模拟"):
    s.running = True

if s.running:
    while not s.is_contained and s.running:
        iterate_step()
        map_ph.pyplot(draw_map())
        
        fig_e, ax_e = plt.subplots(figsize=(5, 3.5))
        if len(s.sse) > 1:
            ax_e.plot(range(1, len(s.sse)+1), s.sse, 'bo-', markersize=4)
            ax_e.axvline(x=s.est_k, color='red', linestyle='--')
            ax_e.set_title("Elbow Method (Source K)")
        elbow_ph.pyplot(fig_e)
        plt.close(fig_e)

        with metric_ph.container():
            st.metric("平均溯源距离误差", f"{s.avg_err:.2f} hops")
            st.metric("算法推断源头数", s.est_k)
            tp = len(s.infected & s.quarantined)
            fn = len(s.infected - s.quarantined)
            st.metric("隔离捕捉率 (Recall)", f"{tp/(tp+fn+1e-5):.1%}")

        plt.close('all')
        time.sleep(0.01)
else:
    map_ph.pyplot(draw_map())
