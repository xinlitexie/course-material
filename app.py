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
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 核心：带地理分区的布局生成 ---
@st.cache_resource
def get_partitioned_network(n, m, k_clusters, seed):
    # 生成基础图
    g = nx.barabasi_albert_graph(n, m, seed=seed)
    
    # --- 布局优化：为每个分区设定一个中心点，强制节点向中心聚集 ---
    np.random.seed(seed)
    centers = np.random.uniform(-1, 1, (k_clusters, 2))
    pos = {}
    nodes = list(g.nodes())
    # 将节点分配给最近的中心
    for i, node in enumerate(nodes):
        c_idx = i % k_clusters
        # 在中心点周围随机扰动
        pos[node] = centers[c_idx] + np.random.normal(0, 0.15, 2)
    
    # 运行物理仿真优化，但保持中心倾向
    pos = nx.spring_layout(g, pos=pos, k=0.1, iterations=20, fixed=None, seed=seed)
    
    # 清理任何可能出现的 NaN (防止绘图崩溃)
    for node in pos:
        if np.any(np.isnan(pos[node])):
            pos[node] = np.array([0.0, 0.0])
            
    return g, pos

# --- 3. 侧边栏：大范围参数 ---
st.sidebar.header("⚙️ 模拟参数")
n_val = st.sidebar.slider("总人口 (N)", 500, 3000, 1500)
m_val = st.sidebar.slider("网络密度 (M)", 1, 3, 2)
t_val = st.sidebar.slider("传播率 (T)", 0.01, 0.4, 0.15)
x_rounds = st.sidebar.slider("干预启动步数 (X)", 1, 30, 8)
src_count = st.sidebar.slider("初始源头数量", 1, 8, 4)
det_val = st.sidebar.slider("流调检测率", 0.1, 1.0, 0.7)
seed_val = st.sidebar.number_input("随机种子", value=42)

# 重置逻辑
params_key = f"{n_val}-{m_val}-{seed_val}-{src_count}"
if 'last_key' not in st.session_state or st.session_state.last_key != params_key:
    st.session_state.last_key = params_key
    g, pos = get_partitioned_network(n_val, m_val, src_count, seed_val)
    np.random.seed(seed_val)
    # 初始源头：从每个地理分区中选一个
    true_sources = []
    for i in range(src_count):
        true_sources.append(list(g.nodes())[i])
        
    st.session_state.update({
        'step': 0, 'running': False, 'g': g, 'pos': pos,
        'true_sources': true_sources,
        'infected': set(true_sources),
        'quarantined': set(),
        'known_cases': set(),
        'node_src_map': {n: i for i, n in enumerate(true_sources)},
        'pred_sources': [], 'est_k': 0, 'avg_err': 0.0, 'is_contained': False
    })

s = st.session_state

# --- 4. 同步演化逻辑 ---
def iterate_step():
    if s.is_contained: return
    
    # A. 同步扩散
    current_active = [n for n in s.infected if n not in s.quarantined]
    if not current_active and s.step > 0:
        s.is_contained = True; return

    new_inf = {}
    for u in current_active:
        u_src = s.node_src_map.get(u, 0)
        for v in s.g.neighbors(u):
            if v not in s.infected and v not in s.quarantined:
                if np.random.rand() < t_val:
                    new_inf[v] = u_src
    
    for v, sid in new_inf.items():
        s.infected.add(v)
        s.node_src_map[v] = sid

    # B. 干预 (Step >= X)
    if s.step >= x_rounds:
        detected = {n for n in (s.infected - s.quarantined) if np.random.rand() < det_val}
        s.known_cases.update(detected)
        
        if len(s.known_cases) >= 5:
            coords = np.array([s.pos[n] for n in s.known_cases])
            k_range = range(1, min(len(s.known_cases), 12))
            sse = [KMeans(n_clusters=k, n_init=3, random_state=42).fit(coords).inertia_ for k in k_range]
            
            # 推断 K 值
            if len(sse) >= 3:
                dists = [np.abs(np.cross(np.array([len(sse),sse[-1]])-np.array([1,sse[0]]), np.array([1,sse[0]])-np.array([i+1,s]))) for i,s in enumerate(sse)]
                s.est_k = k_range[np.argmax(dists)]
                
                km = KMeans(n_clusters=s.est_k, n_init=5).fit(coords)
                s.pred_sources = []
                for i in range(s.est_k):
                    nodes = [list(s.known_cases)[j] for j, label in enumerate(km.labels_) if label == i]
                    if nodes:
                        try:
                            # 局部封控路径
                            st_tree = nx.algorithms.approximation.steiner_tree(s.g, nodes)
                            s.quarantined.update(st_tree.nodes())
                            sub = st_tree if nx.is_connected(st_tree) else st_tree.subgraph(max(nx.connected_components(st_tree), key=len))
                            s.pred_sources.append(min(nx.eccentricity(sub), key=nx.eccentricity(sub).get))
                        except: pass
            
            # 误差计算
            if s.pred_sources:
                errs = [min([nx.shortest_path_length(s.g, ts, ps) for ps in s.pred_sources]) for ts in s.true_sources]
                s.avg_err = np.mean(errs)
    s.step += 1

# --- 5. 渲染引擎：等高线热力图 ---
def draw_map():
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#f8f9fb')
    
    # 基础坐标系范围
    all_coords = np.array(list(s.pos.values()))
    x_min, x_max = all_coords[:,0].min()-0.1, all_coords[:,0].max()+0.1
    y_min, y_max = all_coords[:,1].min()-0.1, all_coords[:,1].max()+0.1

    # A. 渲染 KDE 热力等高线 (核心分区视觉)
    active_inf = list(s.infected - s.quarantined)
    if len(active_inf) >= 5:
        try:
            inf_coords = np.array([s.pos[n] for n in active_inf])
            xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
            kernel = gaussian_kde(inf_coords.T)
            f = np.reshape(kernel(np.vstack([xx.ravel(), yy.ravel()])).T, xx.shape)
            
            # 绘制平滑填充 (Spectral色调类似参考图)
            ax.contourf(xx, yy, f, cmap='Spectral_r', alpha=0.4, levels=15, zorder=0)
            # 绘制深色等高线
            ax.contour(xx, yy, f, colors='black', alpha=0.1, levels=10, linewidths=0.5, zorder=1)
        except: pass

    # B. 绘制极淡的连边层 (防止报错加固)
    if len(s.g.edges()) > 0:
        nx.draw_networkx_edges(s.g, s.pos, ax=ax, alpha=0.01, edge_color='#999999', zorder=2)
    
    # C. 节点分层渲染
    # 1. 健康自由人
    h_free = list(set(s.g.nodes()) - s.infected - s.quarantined)
    nx.draw_networkx_nodes(s.g, s.pos, nodelist=h_free, node_color='#87CEEB', node_size=15, alpha=0.4, ax=ax, zorder=3)

    # 2. 隔离状态 (灰色光晕 + 内部核心)
    if s.quarantined:
        nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(s.quarantined), 
                               node_color='#333333', node_size=100, alpha=0.15, ax=ax, zorder=4)

    # 3. 感染者 (按源头 ID 染色)
    inf_free = list(s.infected - s.quarantined)
    inf_q = list(s.infected & s.quarantined)
    cmap = plt.cm.get_cmap('tab10', max(10, src_count + 1))
    
    for nodes, is_q in [(inf_free, False), (inf_q, True)]:
        if nodes:
            c_list = [cmap(s.node_src_map.get(n, 0)) for n in nodes]
            nx.draw_networkx_nodes(s.g, s.pos, nodelist=nodes, 
                                   node_color=c_list, node_size=35 if is_q else 25,
                                   edgecolors='black' if is_q else 'white', 
                                   linewidths=1.0 if is_q else 0.5, ax=ax, zorder=5)

    # D. 标注源头
    for ts in s.true_sources:
        ax.scatter(s.pos[ts][0], s.pos[ts][1], marker='*', s=450, c='gold', edgecolors='black', zorder=100)
    for ps in s.pred_sources:
        ax.scatter(s.pos[ps][0], s.pos[ps][1], marker='x', s=250, c='purple', linewidths=3, zorder=101)

    ax.set_title(f"Step {s.step} | Inferred K: {s.est_k} | Error: {s.avg_err:.2f} hops", fontsize=14)
    ax.axis('off')
    return fig

# --- 6. UI 主界面 ---
st.title("🛡️ 多源头分区传播动态监测与热力等高线仿真")
st.caption(f"模拟规则：{src_count}个中心点同步爆发 | 第 {x_rounds} 步启动干预 | 节点数 {n_val}")

c1, c2 = st.columns([4, 1.2])
map_spot = c1.empty()
metric_spot = c2.empty()

if st.sidebar.button("▶️ 启动/继续演化"):
    s.running = True

if s.running:
    while not s.is_contained and s.running:
        iterate_step()
        f = draw_map()
        map_spot.pyplot(f)
        plt.close(f) # 关键：释放内存，防止崩溃
        
        with metric_spot.container():
            st.metric("平均溯源距离误差", f"{s.avg_err:.2f} Hops")
            tp = len(s.infected & s.quarantined)
            fn = len(s.infected - s.quarantined)
            st.metric("疫情捕捉率 (Recall)", f"{tp/(tp+fn+1e-5):.1%}")
            st.write(f"**当前状态:** {'封控中' if s.step >= x_rounds else '自由扩散期'}")
        
        time.sleep(0.01)
else:
    map_spot.pyplot(draw_map())
