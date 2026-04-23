import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.cluster import KMeans

# --- 1. 页面配置 ---
st.set_page_config(page_title="多源图传播-精准时序演化系统", layout="wide")
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 网络生成 ---
@st.cache_resource
def get_network(n, m, seed):
    g = nx.barabasi_albert_graph(n, m, seed=seed)
    pos = nx.spring_layout(g, k=0.3, iterations=50, seed=seed)
    return g, pos

# --- 3. 侧边栏：实验参数 ---
st.sidebar.header("⚙️ 模拟参数设置")
n_val = st.sidebar.slider("总人口 (N)", 200, 1200, 700)
m_val = st.sidebar.slider("网络稠密度 (M)", 1, 3, 2)
t_val = st.sidebar.slider("传播概率 (T)", 0.01, 0.5, 0.12)
x_rounds = st.sidebar.slider("干预启动轮次 (X)", 1, 20, 5, help="在第 X 轮之前，系统只传播，不隔离")
det_val = st.sidebar.slider("流调监测率 (Recall)", 0.1, 1.0, 0.7)
src_count = st.sidebar.slider("初始源头数量", 1, 5, 3)
seed_val = st.sidebar.number_input("随机种子", value=42)

# 自动重置逻辑
params_key = f"{n_val}-{m_val}-{seed_val}-{src_count}"
if 'last_key' not in st.session_state or st.session_state.last_key != params_key:
    st.session_state.last_key = params_key
    g, pos = get_network(n_val, m_val, seed_val)
    np.random.seed(seed_val)
    true_sources = list(np.random.choice(list(g.nodes()), src_count, replace=False))
    st.session_state.update({
        'step': 0, 'running': False, 'g': g, 'pos': pos,
        'true_sources': true_sources,
        'infected': set(true_sources),
        'quarantined': set(),
        'known_cases': set(),
        'pred_sources': [],
        'node_source_map': {node: i for i, node in enumerate(true_sources)},
        'sse_list': [], 'est_k': 0, 'is_contained': False
    })

s = st.session_state

# --- 4. 核心演化引擎 ---
def iterate_step():
    if s.is_contained: return
    
    # --- A. 传播阶段 (同步扩散) ---
    # 找到本轮可以传播的人（已感染且未被隔离）
    spreaders = [n for n in s.infected if n not in s.quarantined]
    if not spreaders and s.step > 0:
        s.is_contained = True; return

    newly_infected_this_round = {}
    for u in spreaders:
        u_src_id = s.node_source_map.get(u, 0)
        for v in s.g.neighbors(u):
            # 只有健康且未被隔离的人会被感染
            if v not in s.infected and v not in s.quarantined:
                if np.random.rand() < t_val:
                    newly_infected_this_round[v] = u_src_id
    
    # 统一更新感染状态
    for v, src_id in newly_infected_this_round.items():
        s.infected.add(v)
        s.node_source_map[v] = src_id

    # --- B. 干预阶段 (只有在 step >= x_rounds 时启动) ---
    if s.step >= x_rounds:
        # 1. 监测新病例
        detected = {n for n in (s.infected - s.quarantined) if np.random.rand() < det_val}
        s.known_cases.update(detected)

        # 2. 肘部算法与溯源
        if len(s.known_cases) >= 3:
            coords = np.array([s.pos[n] for n in s.known_cases])
            k_range = range(1, min(len(s.known_cases), 10))
            sse = [KMeans(n_clusters=k, n_init=5, random_state=42).fit(coords).inertia_ for k in k_range]
            s.sse_list = sse
            
            # 动态计算 K (寻找 SSE 曲线拐点)
            if len(sse) >= 3:
                p1, p2 = np.array([1, sse[0]]), np.array([len(sse), sse[-1]])
                dists = [np.abs(np.cross(p2-p1, p1-np.array([i+1, sse[i]]))) / np.linalg.norm(p2-p1) for i in range(len(sse))]
                s.est_k = k_range[np.argmax(dists)]
                
                # 执行聚类隔离
                km = KMeans(n_clusters=s.est_k, n_init=10).fit(coords)
                s.pred_sources = []
                for i in range(s.est_k):
                    cluster_nodes = [list(s.known_cases)[j] for j, label in enumerate(km.labels_) if label == i]
                    if cluster_nodes:
                        try:
                            # 模拟区域封控：Steiner Tree 连接簇内所有发现的病例
                            st_tree = nx.algorithms.approximation.steiner_tree(s.g, cluster_nodes)
                            s.quarantined.update(st_tree.nodes())
                            # 寻找预测源头
                            sub = st_tree if nx.is_connected(st_tree) else st_tree.subgraph(max(nx.connected_components(st_tree), key=len))
                            s.pred_sources.append(min(nx.eccentricity(sub), key=nx.eccentricity(sub).get))
                        except: pass
    
    s.step += 1

# --- 5. 增强可视化 ---
def draw_map():
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw_networkx_edges(s.g, s.pos, ax=ax, alpha=0.03, edge_color='gray')

    inf_free = s.infected - s.quarantined
    inf_q = s.infected & s.quarantined
    healthy_free = set(s.g.nodes()) - s.infected - s.quarantined
    healthy_q = s.quarantined - s.infected

    # 绘制隔离光晕 (灰色外圈)
    if s.quarantined:
        nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(s.quarantined), 
                               node_color='lightgray', node_size=180, alpha=0.4, ax=ax)

    # 绘制健康节点核心 (蓝色)
    nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(healthy_free), 
                           node_color='#87CEEB', node_size=40, alpha=0.7, ax=ax)
    if healthy_q:
        nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(healthy_q), 
                               node_color='#87CEEB', node_size=40, edgecolors='#333333', linewidths=1.5, ax=ax)

    # 绘制感染节点核心 (按源头 ID 染色)
    cmap = plt.cm.get_cmap('Set1', src_count + 1)
    for nodes, is_q in [(inf_free, False), (inf_q, True)]:
        if nodes:
            colors = [cmap(s.node_source_map.get(n, 0)) for n in nodes]
            nx.draw_networkx_nodes(s.g, s.pos, nodelist=list(nodes), 
                                   node_color=colors, node_size=80 if is_q else 60,
                                   edgecolors='white' if not is_q else 'black', 
                                   linewidths=1.5 if is_q else 0.5, ax=ax)

    # 真实源头 (金星) 与 预测源头 (紫叉)
    for ts in s.true_sources:
        ax.scatter(s.pos[ts][0], s.pos[ts][1], marker='*', s=500, c='gold', edgecolors='black', zorder=50)
    for ps in s.pred_sources:
        ax.scatter(s.pos[ps][0], s.pos[ps][1], marker='x', s=300, c='purple', linewidths=4, zorder=51)

    status_txt = "Intervention ACTIVE" if s.step >= x_rounds else "Spread Phase (Intervention LOCKED)"
    ax.set_title(f"STEP {s.step} | {status_txt}", fontsize=14, color='red' if s.step < x_rounds else 'green')
    ax.axis('off')
    return fig

def draw_elbow():
    fig, ax = plt.subplots(figsize=(5, 3.5))
    if len(s.sse_list) > 1:
        ax.plot(range(1, len(s.sse_list)+1), s.sse_list, 'ro-', linewidth=2)
        ax.axvline(x=s.est_k, color='blue', linestyle='--', label=f'Inferred K={s.est_k}')
        ax.set_title("Elbow Method Analysis", fontsize=10)
        ax.set_xlabel("Potential Sources (K)")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "Waiting for Round X...", ha='center')
    plt.tight_layout()
    return fig

# --- 6. 渲染界面 ---
st.title("🛡️ 多源头图传播：精准演化与延迟干预仿真")
st.info(f"💡 规则说明：前 {x_rounds} 步病毒将自由扩散（同步更新），之后系统启动流调、肘部算法溯源并对受影响区域实施隔离。")

c_main, c_side = st.columns([3.5, 1.2])
map_spot = c_main.empty()
elbow_spot = c_side.empty()
matrix_spot = c_side.empty()

if st.sidebar.button("▶️ 开启实时演化模拟"):
    s.running = True

if s.running:
    while not s.is_contained and s.running:
        iterate_step()
        f_map = draw_map()
        f_elbow = draw_elbow()
        
        map_spot.pyplot(f_map)
        elbow_spot.pyplot(f_elbow)
        plt.close(f_map); plt.close(f_elbow)

        with matrix_spot.container():
            st.markdown("#### 📊 实时混淆矩阵")
            tp = len(s.infected & s.quarantined)
            fp = len(s.quarantined - s.infected)
            fn = len(s.infected - s.quarantined)
            tn = n_val - len(s.infected | s.quarantined)
            st.table(pd.DataFrame({"Quarantined": [tp, fp], "Free": [fn, tn]}, index=["Infected", "Healthy"]))
            st.metric("Inferred Sources", s.est_k)
            st.metric("Recall (Catch Rate)", f"{tp/(tp+fn+1e-5):.1%}")

        if s.is_contained:
            st.success("🎉 疫情已得到全面控制！")
            s.running = False
        time.sleep(0.05)
else:
    map_spot.pyplot(draw_map())
    elbow_spot.pyplot(draw_elbow())
