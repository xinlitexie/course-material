import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans

# --- 1. 页面配置 ---
st.set_page_config(page_title="大规模多源传播溯源仿真系统", layout="wide")
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 大规模网络生成优化 ---
@st.cache_resource
def get_large_network(n, m, seed):
    # 使用更高效的生成算法
    g = nx.barabasi_albert_graph(n, m, seed=seed)
    # 针对大规模图优化布局计算
    pos = nx.spring_layout(g, k=1/np.sqrt(n)*2, iterations=30, seed=seed)
    return g, pos

# --- 3. 侧边栏：扩展参数范围 ---
st.sidebar.header("⚙️ 实验参数设置")
n_val = st.sidebar.slider("总人口 (N)", 500, 3000, 1200) # 增加到3000
m_val = st.sidebar.slider("网络稠密度 (M)", 1, 5, 2)
t_val = st.sidebar.slider("传播概率 (T)", 0.01, 0.5, 0.12)
x_rounds = st.sidebar.slider("干预启动轮次 (X)", 1, 30, 8) 
det_val = st.sidebar.slider("流调监测率 (Recall)", 0.1, 1.0, 0.7)
src_count = st.sidebar.slider("初始源头数量", 1, 8, 3) # 增加到8个源头
seed_val = st.sidebar.number_input("随机种子", value=42)

# 参数自动重置
params_key = f"{n_val}-{m_val}-{seed_val}-{src_count}"
if 'last_key' not in st.session_state or st.session_state.last_key != params_key:
    st.session_state.last_key = params_key
    g, pos = get_large_network(n_val, m_val, seed_val)
    np.random.seed(seed_val)
    true_sources = list(np.random.choice(list(g.nodes()), src_count, replace=False))
    st.session_state.update({
        'step': 0, 'running': False, 'g': g, 'pos': pos,
        'true_sources': true_sources,
        'infected': set(true_sources),
        'quarantined': set(),
        'known_cases': set(),
        'pred_sources': [],
        'avg_error_dist': 0.0,
        'node_source_map': {node: i for i, node in enumerate(true_sources)},
        'sse_list': [], 'est_k': 0, 'is_contained': False
    })

s = st.session_state

# --- 4. 核心计算逻辑 ---
def calculate_error_distance():
    """计算多源头平均误差距离：每个真实源头到最近预测源头的平均跳数"""
    if not s.pred_sources: return 0.0
    total_dist = 0
    for ts in s.true_sources:
        dists = []
        for ps in s.pred_sources:
            try:
                # 计算图上的最短路径跳数
                d = nx.shortest_path_length(s.g, source=ts, target=ps)
                dists.append(d)
            except nx.NetworkXNoPath:
                dists.append(20) # 无法到达则设定一个较大的惩罚值
        total_dist += min(dists) if dists else 20
    return total_dist / len(s.true_sources)

def iterate_step():
    if s.is_contained: return
    
    # A. 同步传播 (T 概率)
    spreaders = [n for n in s.infected if n not in s.quarantined]
    if not spreaders and s.step > 0:
        s.is_contained = True; return

    new_inf = {}
    for u in spreaders:
        u_src = s.node_source_map.get(u, 0)
        for v in s.g.neighbors(u):
            if v not in s.infected and v not in s.quarantined:
                if np.random.rand() < t_val:
                    new_inf[v] = u_src
    
    for v, src_id in new_inf.items():
        s.infected.add(v)
        s.node_source_map[v] = src_id

    # B. 干预逻辑 (Step >= X)
    if s.step >= x_rounds:
        detected = {n for n in (s.infected - s.quarantined) if np.random.rand() < det_val}
        s.known_cases.update(detected)

        if len(s.known_cases) >= 5:
            coords = np.array([s.pos[n] for n in s.known_cases])
            k_range = range(1, min(len(s.known_cases), 12))
            sse = [KMeans(n_clusters=k, n_init=5, random_state=42).fit(coords).inertia_ for k in k_range]
            s.sse_list = sse
            
            # 肘部算法推断 K
            if len(sse) >= 3:
                p1, p2 = np.array([1, sse[0]]), np.array([len(sse), sse[-1]])
                dists = [np.abs(np.cross(p2-p1, p1-np.array([i+1, sse[i]]))) / np.linalg.norm(p2-p1) for i in range(len(sse))]
                s.est_k = k_range[np.argmax(dists)]
                
                # 聚类隔离与溯源
                km = KMeans(n_clusters=s.est_k, n_init=10).fit(coords)
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
            
            # 计算溯源误差
            s.avg_error_dist = calculate_error_distance()
    
    s.step += 1

# --- 5. 绘图：针对大规模图优化 ---
def draw_map():
    # 针对大规模节点调小画布点的大小
    fig, ax = plt.subplots(figsize=(10, 8))
    # 大规模下将边设为几乎不可见，避免发丝效应
    nx.draw_networkx_edges(s.g, s.pos, ax=ax, alpha=0.01, edge_color='gray')

    inf_free = s.infected - s.quarantined
    inf_q = s.infected & s.quarantined
    healthy_free = set(s.g.nodes()) - s.infected - s.quarantined
    healthy_q = s.quarantined - s.infected

    # 绘制隔离光晕
    if s.quarantined:
        nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(s.quarantined), 
                               node_color='lightgray', node_size=120, alpha=0.3, ax=ax)

    # 绘制健康节点核心 (缩小尺寸)
    nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(healthy_free), 
                           node_color='#87CEEB', node_size=15, alpha=0.6, ax=ax)
    if healthy_q:
        nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(healthy_q), 
                               node_color='#87CEEB', node_size=15, edgecolors='black', linewidths=1, ax=ax)

    # 绘制感染节点核心
    cmap = plt.cm.get_cmap('tab10', src_count + 1)
    for nodes, is_q in [(inf_free, False), (inf_q, True)]:
        if nodes:
            colors = [cmap(s.node_source_map.get(n, 0)) for n in nodes]
            nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(nodes), 
                                   node_color=colors, node_size=40 if is_q else 30,
                                   edgecolors='white' if not is_q else 'black', 
                                   linewidths=1 if is_q else 0.5, ax=ax)

    # 真实源头 (金星) 与 预测源头 (紫叉)
    for ts in s.true_sources:
        ax.scatter(s.pos[ts][0], s.pos[ts][1], marker='*', s=400, c='gold', edgecolors='black', zorder=50)
    for ps in s.pred_sources:
        ax.scatter(s.pos[ps][0], s.pos[ps][1], marker='x', s=250, c='purple', linewidths=3, zorder=51)

    ax.set_title(f"STEP {s.step} | N={n_val} | Error Distance: {s.avg_error_dist:.2f} hops", fontsize=12)
    ax.axis('off')
    return fig

def draw_elbow():
    fig, ax = plt.subplots(figsize=(5, 3))
    if len(s.sse_list) > 1:
        ax.plot(range(1, len(s.sse_list)+1), s.sse_list, 'ro-', linewidth=1.5)
        ax.axvline(x=s.est_k, color='blue', linestyle='--', label=f'Best K={s.est_k}')
        ax.set_title("Elbow Method Analysis", fontsize=10)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Waiting for Round X...", ha='center')
    plt.tight_layout()
    return fig

# --- 6. UI 主界面 ---
st.title("🛡️ 大规模多源传播动态仿真与溯源评估系统")
st.markdown(f"""
<div style="background-color:#f8f9fa; padding:10px; border-radius:5px; border-left:5px solid #28a745; margin-bottom:15px">
<b>实验规则：</b> 当前设定为 {src_count} 个同步爆发点。系统在第 {x_rounds} 轮前仅观察传播，随后启动肘部算法自动推断源头数量并实施封控。
</div>
""", unsafe_allow_html=True)

c_left, c_right = st.columns([3.5, 1.2])
map_spot = c_left.empty()
elbow_spot = c_right.empty()
metrics_spot = c_right.empty()

if st.sidebar.button("▶️ 开启实时演化模拟"):
    s.running = True

if s.running:
    while not s.is_contained and s.running:
        iterate_step()
        map_spot.pyplot(draw_map())
        elbow_spot.pyplot(draw_elbow())
        
        with metrics_spot.container():
            st.markdown("#### 📊 实时评估指标")
            st.metric("平均溯源误差 (跳数)", f"{s.avg_error_dist:.2f} hops")
            
            tp = len(s.infected & s.quarantined)
            fn = len(s.infected - s.quarantined)
            st.metric("疫情捕捉率 (Recall)", f"{tp/(tp+fn+1e-5):.1%}")
            
            st.write("---")
            st.write("**混淆矩阵**")
            fp = len(s.quarantined - s.infected)
            tn = n_val - len(s.infected | s.quarantined)
            st.table(pd.DataFrame({"Isolated": [tp, fp], "Free": [fn, tn]}, index=["Infected", "Healthy"]))

        if s.is_contained:
            st.success("🎉 传播链已全部切断！")
            s.running = False
        time.sleep(0.01) # 大规模图计算较慢，此处延迟设低
else:
    map_spot.pyplot(draw_map())
    elbow_spot.pyplot(draw_elbow())
