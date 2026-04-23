import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans

# --- 1. 页面配置 ---
st.set_page_config(page_title="多源动态溯源可视化系统", layout="wide")
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 缓存网络生成 ---
@st.cache_resource
def get_network(n, m, seed):
    g = nx.barabasi_albert_graph(n, m, seed=seed)
    pos = nx.spring_layout(g, k=0.3, iterations=50, seed=seed)
    return g, pos

# --- 3. 侧边栏与参数重置 ---
st.sidebar.header("🕹️ 模拟控制")
n_val = st.sidebar.slider("总人口", 200, 1000, 600)
m_val = st.sidebar.slider("网络密度", 1, 3, 2)
t_val = st.sidebar.slider("传播率 (T)", 0.01, 0.4, 0.15)
det_val = st.sidebar.slider("检测率 (Recall)", 0.1, 1.0, 0.7)
source_count = st.sidebar.slider("初始源头数", 1, 5, 3)
seed_val = st.sidebar.number_input("随机种子", value=42)

# 参数变动自动重置
params_key = f"{n_val}-{m_val}-{seed_val}-{source_count}"
if 'last_key' not in st.session_state or st.session_state.last_key != params_key:
    st.session_state.last_key = params_key
    g, pos = get_network(n_val, m_val, seed_val)
    np.random.seed(seed_val)
    true_sources = list(np.random.choice(list(g.nodes()), source_count, replace=False))
    
    st.session_state.update({
        'step': 0, 'running': False, 'g': g, 'pos': pos,
        'true_sources': true_sources,
        'infected': set(true_sources),
        'quarantined': set(),
        'known_cases': set(),
        'pred_sources': [],
        'node_source_map': {s: i for i, s in enumerate(true_sources)}, # 记录每个感染者的来源
        'sse_list': [],
        'est_k': 1,
        'is_contained': False
    })

s = st.session_state

# --- 4. 核心逻辑：演化与算法 ---
def iterate_step():
    if s.is_contained: return
    
    # A. 传播逻辑：带源头 ID 继承
    active_spreaders = list(s.infected - s.quarantined)
    if not active_spreaders and s.step > 0:
        s.is_contained = True; return

    new_infections = {}
    for u in active_spreaders:
        u_src = s.node_source_map.get(u, 0)
        for v in s.g.neighbors(u):
            if v not in s.infected and v not in s.quarantined:
                if np.random.rand() < t_val:
                    new_infections[v] = u_src
    
    for v, src_id in new_infections.items():
        s.infected.add(v)
        s.node_source_map[v] = src_id

    # B. 流调监测
    detected = {n for n in (s.infected - s.quarantined) if np.random.rand() < det_val}
    s.known_cases.update(detected)

    # C. 肘部算法溯源
    if len(s.known_cases) >= 3:
        coords = np.array([s.pos[n] for n in s.known_cases])
        k_range = range(1, min(len(s.known_cases), 9))
        sse = [KMeans(n_clusters=k, n_init=5, random_state=42).fit(coords).inertia_ for k in k_range]
        s.sse_list = sse
        
        if len(sse) >= 3:
            # 几何拐点检测
            p1, p2 = np.array([1, sse[0]]), np.array([len(sse), sse[-1]])
            dists = [np.abs(np.cross(p2-p1, p1-np.array([i+1, sse[i]]))) / np.linalg.norm(p2-p1) for i in range(len(sse))]
            s.est_k = k_range[np.argmax(dists)]
            
            km = KMeans(n_clusters=s.est_k, n_init=10).fit(coords)
            s.pred_sources = []
            for i in range(s.est_k):
                cluster = [list(s.known_cases)[j] for j, label in enumerate(km.labels_) if label == i]
                if cluster:
                    try:
                        st_tree = nx.algorithms.approximation.steiner_tree(s.g, cluster)
                        s.quarantined.update(st_tree.nodes())
                        sub = st_tree if nx.is_connected(st_tree) else st_tree.subgraph(max(nx.connected_components(st_tree), key=len))
                        s.pred_sources.append(min(nx.eccentricity(sub), key=nx.eccentricity(sub).get))
                    except: pass
    s.step += 1

# --- 5. 增强型绘图逻辑 ---
def draw_map():
    fig, ax = plt.subplots(figsize=(11, 9))
    nx.draw_networkx_edges(s.g, s.pos, ax=ax, alpha=0.03, edge_color='gray')

    # 节点分类
    all_nodes = set(s.g.nodes())
    inf_nodes = s.infected
    q_nodes = s.quarantined
    
    healthy_free = all_nodes - inf_nodes - q_nodes
    inf_free = inf_nodes - q_nodes
    healthy_q = q_nodes - inf_nodes
    inf_q = inf_nodes & q_nodes

    # 1. 先画底层：如果是隔离节点，先画一层大的灰色光晕
    if q_nodes:
        nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(q_nodes), 
                               node_color='lightgray', node_size=180, alpha=0.5, ax=ax)

    # 2. 画健康节点（自由+隔离）
    nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(healthy_free), 
                           node_color='#87CEEB', node_size=40, alpha=0.7, ax=ax)
    if healthy_q:
        nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(healthy_q), 
                               node_color='#87CEEB', node_size=40, edgecolors='#444444', linewidths=2, ax=ax)

    # 3. 画感染节点（自由+隔离）- 根据 Source ID 着色
    cmap = plt.cm.get_cmap('Set1', source_count + 1)
    for node_list, is_q in [(inf_free, False), (inf_q, True)]:
        if node_list:
            node_colors = [cmap(s.node_source_map.get(n, 0)) for n in node_list]
            nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(node_list), 
                                   node_color=node_colors, node_size=80 if is_q else 60,
                                   edgecolors='white' if not is_q else '#222222', 
                                   linewidths=1.5 if is_q else 0.5, ax=ax)

    # 4. 标注源头
    for ts in s.true_sources:
        ax.scatter(s.pos[ts][0], s.pos[ts][1], marker='*', s=600, c='gold', edgecolors='black', zorder=50)
    for ps in s.pred_sources:
        ax.scatter(s.pos[ps][0], s.pos[ps][1], marker='x', s=350, c='purple', linewidths=4, zorder=51)

    ax.set_title(f"LIVE EVOLUTION STEP: {s.step}\n(Large Gray Ring = Quarantined | Colored by Infection Source)", fontsize=14)
    ax.axis('off')
    return fig

def draw_elbow():
    fig, ax = plt.subplots(figsize=(5, 3.5))
    if len(s.sse_list) > 1:
        ax.plot(range(1, len(s.sse_list)+1), s.sse_list, 'ro-', linewidth=2, markersize=5)
        ax.axvline(x=s.est_k, color='blue', linestyle='--', label=f'Best K={s.est_k}')
        ax.set_title("Elbow Method (Source Inference)", fontsize=10)
        ax.set_ylabel("SSE")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Collecting CDC Data...", ha='center')
    plt.tight_layout()
    return fig

# --- 6. 布局渲染 ---
st.title("🛡️ 多源头图传播：实时演化、肘部算法溯源与分区隔离监控")

col_left, col_right = st.columns([3.5, 1.2])
map_place = col_left.empty()
elbow_place = col_right.empty()
matrix_place = col_right.empty()

if st.sidebar.button("▶️ 启动实时模拟演化"):
    s.running = True

if s.running:
    while not s.is_contained and s.running:
        iterate_step()
        f_map = draw_map()
        f_elbow = draw_elbow()
        
        map_place.pyplot(f_map)
        elbow_place.pyplot(f_elbow)
        plt.close(f_map)
        plt.close(f_elbow)

        with matrix_place.container():
            st.markdown("#### 📊 实时混淆矩阵")
            tp = len(s.infected & s.quarantined)
            fp = len(s.quarantined - s.infected)
            fn = len(s.infected - s.quarantined)
            tn = n_val - len(s.infected | s.quarantined)
            
            df = pd.DataFrame({
                "In Quarantine": [f"TP: {tp}", f"FP: {fp}"],
                "Active/Free": [f"FN: {fn}", f"TN: {tn}"]
            }, index=["Infected (Sick)", "Healthy (Safe)"])
            st.table(df)
            st.metric("Source Detection K", s.est_k)
            st.metric("Infection Recall", f"{tp/(tp+fn+1e-5):.1%}")

        if s.is_contained:
            st.balloons()
            s.running = False
        time.sleep(0.05)
else:
    map_place.pyplot(draw_map())
    elbow_place.pyplot(draw_elbow())
    matrix_place.info("调整参数将重置。点击侧边栏按钮开启演化。")
