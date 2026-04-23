import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans

# --- 1. 页面配置 ---
st.set_page_config(page_title="多源图传播实时系统", layout="wide")

# --- 2. 缓存网络生成 ---
@st.cache_resource
def get_network(n, m, seed):
    g = nx.barabasi_albert_graph(n, m, seed=seed)
    pos = nx.spring_layout(g, k=0.3, iterations=50, seed=seed)
    return g, pos

# --- 3. 侧边栏与参数重置逻辑 ---
st.sidebar.header("🕹️ 控制面板")
n_val = st.sidebar.slider("总人口", 200, 1000, 600)
m_val = st.sidebar.slider("网络连接密度", 1, 5, 2)
t_val = st.sidebar.slider("传播率 (T)", 0.01, 0.5, 0.15)
det_val = st.sidebar.slider("检测率 (Recall)", 0.0, 1.0, 0.6)
source_count = st.sidebar.slider("源头数量", 1, 5, 3)
speed = st.sidebar.slider("演化速度 (延迟)", 0.01, 0.5, 0.1)
seed_val = st.sidebar.number_input("随机种子", value=42)

# 参数联动重置
params_key = f"{n_val}-{m_val}-{seed_val}-{source_count}"
if 'last_key' not in st.session_state or st.session_state.last_key != params_key:
    st.session_state.last_key = params_key
    g, pos = get_network(n_val, m_val, seed_val)
    np.random.seed(seed_val)
    # 随机选定初始源头
    true_sources = list(np.random.choice(list(g.nodes()), source_count, replace=False))
    
    st.session_state.update({
        'step': 0, 'running': False, 'g': g, 'pos': pos,
        'true_sources': true_sources,
        'infected': set(true_sources), # 初始只有源头感染
        'quarantined': set(),
        'known_cases': set(),
        'pred_sources': [],
        'node_source_map': {s: i for i, s in enumerate(true_sources)}, # 追踪感染来源
        'is_contained': False
    })

s = st.session_state

# --- 4. 实时传播与溯源逻辑 ---
def iterate_step():
    if s.is_contained: return
    
    # A. 核心传播逻辑：寻找感染者中未被隔离的“漏网之鱼”
    active_spreaders = list(s.infected - s.quarantined)
    if not active_spreaders and s.step > 0:
        s.is_contained = True
        return

    new_infections = {}
    for u in active_spreaders:
        # 获取感染者 u 的来源（用于颜色区分）
        origin_source = s.node_source_map.get(u, 0)
        for v in s.g.neighbors(u):
            if v not in s.infected and v not in s.quarantined:
                if np.random.rand() < t_val:
                    new_infections[v] = origin_source
    
    # 更新感染列表和来源图
    for v, src_id in new_infections.items():
        s.infected.add(v)
        s.node_source_map[v] = src_id

    # B. 流调与溯源隔离
    # 发现病例：只有感染且未隔离的才可能被发现
    newly_found = {n for n in (s.infected - s.quarantined) if np.random.rand() < det_val}
    s.known_cases.update(newly_found)

    if len(s.known_cases) >= source_count * 2:
        coords = np.array([s.pos[node] for node in s.known_cases])
        # 肘部算法检测 K
        ks = range(1, min(len(s.known_cases), 8))
        sse = [KMeans(n_clusters=k, n_init=5).fit(coords).inertia_ for k in ks]
        est_k = ks[np.argmax(np.abs(np.diff(sse, 2)))] if len(sse) > 2 else 1
        
        km = KMeans(n_clusters=est_k, n_init=5).fit(coords)
        s.pred_sources = []
        for i in range(est_k):
            cluster = [list(s.known_cases)[j] for j, label in enumerate(km.labels_) if label == i]
            if cluster:
                try:
                    st_tree = nx.algorithms.approximation.steiner_tree(s.g, cluster)
                    s.quarantined.update(st_tree.nodes())
                    # 寻找簇中心
                    sub = st_tree if nx.is_connected(st_tree) else st_tree.subgraph(max(nx.connected_components(st_tree), key=len))
                    s.pred_sources.append(min(nx.eccentricity(sub), key=nx.eccentricity(sub).get))
                except: pass
    s.step += 1

# --- 5. 绘图：颜色区分与实时展示 ---
def draw_map():
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 1. 基础连边
    nx.draw_networkx_edges(s.g, s.pos, ax=ax, alpha=0.03, edge_color='gray')
    
    # 2. 分类节点
    all_nodes = set(s.g.nodes())
    quarantined = s.quarantined
    infected_active = s.infected - quarantined
    healthy = all_nodes - s.infected - quarantined

    # A. 健康节点：统一天蓝色
    nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(healthy), 
                           node_color='#87CEEB', node_size=30, alpha=0.6, ax=ax)
    
    # B. 隔离节点：深灰色
    if quarantined:
        nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(quarantined), 
                               node_color='#444444', node_size=80, alpha=0.4, ax=ax)

    # C. 感染节点：根据来源源头赋予不同的红色调 (满足“区分源头”)
    if infected_active:
        # 使用不同的调色盘区分源头领地
        colors = [s.node_source_map.get(n, 0) for n in infected_active]
        nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(infected_active), 
                               node_color=colors, cmap=plt.cm.YlOrRd, 
                               vmin=-source_count, vmax=source_count,
                               node_size=70, edgecolors='white', linewidths=0.5, ax=ax)

    # 3. 标注源头
    for ts in s.true_sources:
        ax.scatter(s.pos[ts][0], s.pos[ts][1], marker='*', s=500, 
                   c='gold', edgecolors='black', linewidths=1.5, zorder=20, label="Real Source")
    for ps in s.pred_sources:
        ax.scatter(s.pos[ps][0], s.pos[ps][1], marker='x', s=300, 
                   c='purple', linewidths=4, zorder=21, label="Est Source")

    ax.set_title(f"STEP: {s.step} | RED: INFECTED | GRAY: QUARANTINED | BLUE: HEALTHY", fontsize=15)
    ax.axis('off')
    return fig

# --- 6. 布局渲染 ---
st.title("🛡️ 社交网络多源传播与动态防控实战演练")

col_left, col_right = st.columns([4, 1.5])
map_place = col_left.empty()
stats_place = col_right.empty()

if st.sidebar.button("▶️ 开始模拟演化"):
    s.running = True

if s.running:
    while not s.is_contained and s.running:
        iterate_step()
        f = draw_map()
        map_place.pyplot(f)
        plt.close(f)
        
        with stats_place.container():
            st.write("### 📈 实时统计")
            st.metric("总感染数", len(s.infected))
            st.metric("已隔离人数", len(s.quarantined))
            st.metric("健康人口", n_val - len(s.infected))
            
            st.write("---")
            st.write("#### 混淆矩阵")
            tp = len(s.infected & s.quarantined)
            fn = len(s.infected - s.quarantined)
            st.table(pd.DataFrame({
                "隔离": [f"TP: {tp}", f"FP: {len(s.quarantined - s.infected)}"],
                "自由": [f"FN: {fn}", f"TN: {n_val - len(s.infected | s.quarantined)}"]
            }, index=["感染者", "健康者"]))
            st.progress(tp/(tp+fn+1e-5), text="疫情控制进度")

        if s.is_contained:
            st.success("🎉 疫情已完全控制（传播链被切断）")
            s.running = False
        time.sleep(speed)
else:
    map_place.pyplot(draw_map())
    stats_place.info("调整参数将重置，点击左侧按钮开始演化。")
