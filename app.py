import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# --- 1. 核心逻辑：自动寻找最佳 K 值 (肘部算法) ---
def find_optimal_k(coords, max_k=8):
    """
    使用肘部法则自动确定源头数量
    原理：计算 SSE 曲线的变化率，寻找拐点
    """
    if len(coords) < 3:
        return 1
    
    k_range = range(1, min(len(coords), max_k + 1))
    sse = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=5, random_state=42).fit(coords)
        sse.append(kmeans.inertia_)
    
    if len(sse) < 3:
        return 1
    
    # 肘部检测逻辑：计算每个点到首尾连线的距离，距离最远的点即为肘部
    # 这比单纯看斜率更鲁棒
    p1 = np.array([k_range[0], sse[0]])
    p2 = np.array([k_range[-1], sse[-1]])
    
    distances = []
    for i in range(len(sse)):
        p0 = np.array([k_range[i], sse[i]])
        # 点到直线的距离公式
        d = np.abs(np.cross(p2-p1, p1-p0)) / np.linalg.norm(p2-p1)
        distances.append(d)
        
    return k_range[np.argmax(distances)]

# --- 2. 缓存网络与分区生成 ---
@st.cache_resource
def get_network(n, m, seed):
    # 生成无标度网络
    g = nx.barabasi_albert_graph(n, m, seed=seed)
    # 使用较多迭代次数让布局更稳固
    pos = nx.spring_layout(g, k=0.15, iterations=50, seed=seed)
    
    # 使用 Louvain 社区发现算法进行行政分区
    communities = nx.community.louvain_communities(g, seed=seed)
    partition = {}
    for i, comm in enumerate(communities):
        for node in comm:
            partition[node] = i
    return g, pos, partition

# --- 3. 页面配置 ---
st.set_page_config(page_title="多源溯源肘部算法仿真", layout="wide")
st.sidebar.header("⚙️ 实验设置")

n_val = st.sidebar.slider("节点数量", 300, 1500, 800)
m_val = st.sidebar.slider("稠密度 (M)", 1, 3, 2)
real_source_count = st.sidebar.slider("实际初始源头数", 1, 5, 3)
t_val = st.sidebar.slider("传播概率", 0.05, 0.4, 0.15)
det_val = st.sidebar.slider("流调监测率", 0.1, 1.0, 0.6)
hunt_val = st.sidebar.slider("开始干预步数", 1, 10, 3)

# --- 4. 初始化状态 ---
if 'step' not in st.session_state or st.sidebar.button("🔄 重置并运行"):
    g, pos, partition = get_network(n_val, m_val, 42)
    # 随机选择多个真实源头
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
        'estimated_k': 0,
        'is_contained': False
    })

s = st.session_state

# --- 5. 仿真步进函数 ---
def iterate_step():
    if s.is_contained: return

    # A. 传播过程
    leaked = list(s.infected - s.quarantined)
    if not leaked and s.step > 0:
        s.is_contained = True
        return

    new_inf = set()
    for u in leaked:
        for v in s.g.neighbors(u):
            if v not in s.infected and v not in s.quarantined:
                if np.random.rand() < t_val:
                    new_inf.add(v)
    s.infected.update(new_inf)

    # B. 多源溯源 (带肘部算法)
    if s.step >= hunt_val:
        # 1. 发现新病例
        newly_found = {n for n in (s.infected - s.quarantined) if np.random.rand() < det_val}
        s.known_cases.update(newly_found)
        
        if len(s.known_cases) >= 3:
            # 提取已知病例坐标
            case_list = list(s.known_cases)
            coords = np.array([s.pos[node] for node in case_list])
            
            # 2. 肘部算法检测源头数量
            optimal_k = find_optimal_k(coords)
            s.estimated_k = optimal_k
            
            # 3. 执行聚类
            kmeans = KMeans(n_clusters=optimal_k, n_init=10).fit(coords)
            s.pred_sources = []
            
            # 4. 对每个簇进行局部隔离与溯源
            for i in range(optimal_k):
                cluster_nodes = [case_list[j] for j, label in enumerate(kmeans.labels_) if label == i]
                if len(cluster_nodes) > 0:
                    try:
                        # 局部 Steiner Tree 模拟封控区
                        st_tree = nx.algorithms.approximation.steiner_tree(s.g, cluster_nodes)
                        s.quarantined.update(st_tree.nodes())
                        # 预测源头为局部离心率中心
                        sub = st_tree if nx.is_connected(st_tree) else st_tree.subgraph(max(nx.connected_components(st_tree), key=len))
                        ecc = nx.eccentricity(sub)
                        s.pred_sources.append(min(ecc, key=ecc.get))
                    except: pass
    s.step += 1

# --- 6. 绘图函数 ---
def draw_map():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 状态分类
    inf_alive = s.infected - s.quarantined
    q_nodes = s.quarantined
    clean_nodes = set(s.g.nodes()) - s.infected - s.quarantined
    
    # 1. 绘制连边
    nx.draw_networkx_edges(s.g, s.pos, ax=ax, alpha=0.01, edge_color='gray')
    
    # 2. 绘制分区背景 (根据 partition 分类)
    nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(clean_nodes), 
                           node_color=[s.partition[n] for n in clean_nodes],
                           cmap=plt.cm.Pastel1, node_size=30, alpha=0.7, ax=ax)
    
    # 3. 绘制隔离区
    if q_nodes:
        nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(q_nodes), 
                               node_color='#333333', node_size=120, alpha=0.4, ax=ax)

    # 4. 绘制活跃感染者
    nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(inf_alive), 
                           node_color='red', node_size=60, edgecolors='white', ax=ax)
    
    # 5. 标注真实源头 (金星)
    for ts in s.true_sources:
        ax.scatter(s.pos[ts][0], s.pos[ts][1], marker='*', s=400, c='gold', edgecolors='black', zorder=20)
    
    # 6. 标注预测源头 (紫叉)
    for ps in s.pred_sources:
        ax.scatter(s.pos[ps][0], s.pos[ps][1], marker='x', s=300, c='purple', linewidths=3, zorder=21)

    ax.set_title(f"Step: {s.step} | 推测源头数: {s.estimated_k} | 感染: {len(s.infected)}", fontsize=16)
    ax.axis('off')
    return fig

# --- 7. UI 主体 ---
st.title("🛡️ 多源头动态传播与自动溯源系统")
st.markdown(f"**当前状态：** 真实源头数 `{real_source_count}` | 算法推测源头数 `{s.estimated_k}`")

col1, col2 = st.columns([3, 1])

if st.sidebar.button("▶️ 开始模拟"):
    s.running = True

map_ph = col1.empty()
metrics_ph = col2.empty()
dist_ph = col2.empty()

if s.running:
    while not s.is_contained:
        iterate_step()
        fig = draw_map()
        map_ph.pyplot(fig)
        plt.close(fig)
        
        # 指标更新
        with metrics_ph.container():
            st.metric("累计感染总数", len(s.infected))
            st.metric("当前隔离人数", len(s.quarantined))
            acc = len(s.quarantined & s.infected) / max(1, len(s.infected))
            st.metric("有效隔离率 (Recall)", f"{acc:.1%}")

        # 分区统计
        with dist_ph.container():
            st.write("#### 🏘️ 分区感染实况")
            p_stats = []
            for p_id in range(max(s.partition.values())+1):
                p_nodes = [n for n, p in s.partition.items() if p == p_id]
                inf = [n for n in p_nodes if n in s.infected]
                p_stats.append({"ID": p_id, "感染率": len(inf)/len(p_nodes)})
            
            df = pd.DataFrame(p_stats).sort_values("感染率", ascending=False)
            st.dataframe(df.style.background_gradient(cmap="Reds"), hide_index=True)

        if s.step > 60: s.is_contained = True
        time.sleep(0.05)
else:
    map_ph.pyplot(draw_map())
