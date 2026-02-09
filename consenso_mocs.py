import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# =========================================================
# 1) Construcción de gráficas (topologías)
# =========================================================
def build_topology_graph(N: int, topology: str, seed: int = 7, weighted: bool = True) -> nx.Graph:
    """
    Topologías soportadas (no dirigidas):
      - "chain"     : cadena (path)
      - "ring"      : anillo (cycle)
      - "star"      : estrella tipo hub (centro = 0)
      - "complete"  : red completa
      - "pentagram" : pentagrama (estrella 5 puntas con cruces) -> requiere N=5
    """
    topology = topology.lower().strip()

    if topology == "chain":
        G = nx.path_graph(N)

    elif topology == "ring":
        G = nx.cycle_graph(N)

    elif topology == "star":
        # star_graph(k) crea k+1 nodos => para N nodos, k=N-1
        G = nx.star_graph(N - 1)

    elif topology == "complete":
        G = nx.complete_graph(N)

    elif topology == "pentagram":
        if N != 5:
            raise ValueError('La topología "pentagram" requiere N=5 (5 nodos).')
        G = nx.Graph()
        G.add_nodes_from(range(5))
        # Pentagrama {5/2}: conecta cada nodo con el que está "a dos pasos"
        for k in range(5):
            G.add_edge(k, (k + 2) % 5)

    else:
        raise ValueError("Topologia no soportada. Usa: chain, ring, star, complete, pentagram")

    # Pesos (opcional)
    if weighted:
        rng = np.random.default_rng(seed)
        for (u, v) in G.edges():
            G[u][v]["weight"] = 0.5 + rng.random()  # (0.5, 1.5)
    else:
        for (u, v) in G.edges():
            G[u][v]["weight"] = 1.0

    return G


def build_disconnected_example(N: int, seed: int = 11, weighted: bool = True) -> nx.Graph:
    """
    Ejemplo desconectado claro: dos cadenas separadas (2 componentes).
    """
    n1 = N // 2
    n2 = N - n1
    G1 = nx.path_graph(n1)
    G2 = nx.path_graph(n2)
    G = nx.disjoint_union(G1, G2)

    if weighted:
        rng = np.random.default_rng(seed)
        for (u, v) in G.edges():
            G[u][v]["weight"] = 0.5 + rng.random()
    else:
        for (u, v) in G.edges():
            G[u][v]["weight"] = 1.0

    return G


# =========================================================
# 1.1) Layouts (incluye pentagrama)
# =========================================================
def regular_ngon_layout(nodes, n: int, ang0=np.pi / 2, R: float = 1.0):
    """
    Coloca n nodos en un polígono regular (en círculo) en el orden dado por `nodes`.
    """
    pos = {}
    for i, node in enumerate(nodes):
        ang = ang0 + 2 * np.pi * i / n
        pos[node] = (R * np.cos(ang), R * np.sin(ang))
    return pos


def topology_layout(G: nx.Graph, topology: str, seed: int = 1):
    """
    Layout recomendado por topología (para que se vea bonito).
    """
    topology = topology.lower().strip()

    if topology == "chain":
        return {i: (i, 0.0) for i in G.nodes()}  # en línea

    if topology == "ring":
        return nx.circular_layout(G)

    if topology == "complete":
        return nx.circular_layout(G)

    if topology == "star":
        center = 0
        leaves = [n for n in G.nodes() if n != center]
        return nx.shell_layout(G, nlist=[[center], leaves])

    if topology == "pentagram":
        # Coloca los 5 nodos en un pentágono regular; las aristas (saltando 1) dibujan el pentagrama.
        nodes = sorted(G.nodes())
        return regular_ngon_layout(nodes, n=5, ang0=np.pi/2, R=1.0)

    return nx.spring_layout(G, seed=seed)


# =========================================================
# 2) Laplaciana y espectro (SIN SciPy)
# =========================================================
def laplacian_matrix_undirected(G: nx.Graph, weight: str = "weight") -> np.ndarray:  # 
    """
    Laplaciana L = D - A sin depender de SciPy.
    Usa el orden de nodos: list(G.nodes()).
    """
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    A = np.zeros((N, N), dtype=float)

    for u, v, data in G.edges(data=True):
        w = float(data.get(weight, 1.0))
        i, j = idx[u], idx[v]
        A[i, j] += w
        A[j, i] += w  # no dirigido

    D = np.diag(A.sum(axis=1))
    L = D - A
    return L


def laplacian_spectrum(L: np.ndarray) -> np.ndarray:
    """
    Autovalores de L (ordenados). Para L simétrica usamos eigvalsh.
    """
    eigvals = np.linalg.eigvalsh(L)
    eigvals = np.sort(np.real(eigvals))
    eigvals[np.abs(eigvals) < 1e-10] = 0.0
    return eigvals


def algebraic_connectivity(L: np.ndarray) -> float:
    """
    lambda2 = segundo autovalor más pequeño.
    """
    ev = laplacian_spectrum(L)
    return float(ev[1]) if len(ev) >= 2 else 0.0


def num_connected_components_from_spectrum(L: np.ndarray) -> int:   
    """
    #componentes = multiplicidad del 0 en el espectro de L (grafo no dirigido). 
    """
    ev = laplacian_spectrum(L)
    return int(np.sum(np.isclose(ev, 0.0, atol=1e-10)))


# =========================================================
# 3) Simulación del consenso (Euler)
# =========================================================
def simulate_consensus_euler(L: np.ndarray, x0: np.ndarray, T: float, dt: float):
    steps = int(np.floor(T / dt)) + 1
    N = x0.size
    X = np.zeros((steps, N), dtype=float)
    t = np.linspace(0.0, T, steps)

    x = x0.astype(float).copy()
    X[0] = x

    for k in range(1, steps):
        x = x - dt * (L @ x)
        X[k] = x

    return t, X


def simulate_switching_topology(segments, x0: np.ndarray, dt: float):
    t_all = []
    X_all = []

    x = x0.astype(float).copy()
    t_offset = 0.0

    for (Tseg, Lseg) in segments:
        t, X = simulate_consensus_euler(Lseg, x, Tseg, dt)
        t = t + t_offset

        if len(t_all) > 0:
            t = t[1:]
            X = X[1:]

        t_all.append(t)
        X_all.append(X)

        x = X[-1].copy()
        t_offset = t_all[-1][-1]

    return np.concatenate(t_all), np.vstack(X_all)


# =========================================================
# 4) Visualizaciones
# =========================================================
def plot_network(G: nx.Graph, pos, title: str, filename: str = None, topology: str = ""):
    plt.figure()

    # Para que el pentagrama se parezca más a la imagen: aristas más gruesas
    width = 7.0 if topology == "pentagram" else 2.5

    nx.draw(
        G, pos,
        with_labels=True,
        node_size=700,
        width=width,
        font_size=10
    )

    plt.title(title)
    plt.axis("equal")
    plt.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.08)  

    if filename:
        plt.savefig(filename, dpi=200)
    plt.show()


def plot_states(t: np.ndarray, X: np.ndarray, title: str, filename: str = None):
    plt.figure()
    for i in range(X.shape[1]):
        plt.plot(t, X[:, i], label=f"x{i}")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Estado")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.12)
    if filename:
        plt.savefig(filename, dpi=200)
    plt.show()


# =========================================================
# 5) Main
# =========================================================
def run_case(N: int, dt: float, topology_conn: str, seed_w: int = 7):
    rng = np.random.default_rng(3)
    x0 = rng.normal(loc=0.0, scale=2.0, size=N)

    # --- Conectada (topología elegida)
    G_conn = build_topology_graph(N, topology_conn, seed=seed_w, weighted=True)
    L_conn = laplacian_matrix_undirected(G_conn)
    ev_conn = laplacian_spectrum(L_conn)
    lam2_conn = algebraic_connectivity(L_conn)
    cc_conn = num_connected_components_from_spectrum(L_conn)

    # --- Desconectada (ejemplo)
    G_disc = build_disconnected_example(N, seed=11, weighted=True)
    L_disc = laplacian_matrix_undirected(G_disc)
    ev_disc = laplacian_spectrum(L_disc)
    lam2_disc = algebraic_connectivity(L_disc)
    cc_disc = num_connected_components_from_spectrum(L_disc)

    # Layouts
    pos_conn = topology_layout(G_conn, topology_conn, seed=1)
    pos_disc = nx.spring_layout(G_disc, seed=2)

    # Dibujo de redes
    plot_network(
        G_conn, pos_conn,
        f"Red conectada ({topology_conn}) | comp={cc_conn} | lambda2≈{lam2_conn:.4f}",
        f"red_conectada_{topology_conn}.png",
        topology=topology_conn
    )
    plot_network(
        G_disc, pos_disc,
        f"Red desconectada (2 cadenas) | comp={cc_disc} | lambda2≈{lam2_disc:.4f}",
        "red_desconectada.png",
        topology=""
    )

    # Espectro
    print(f"\n=== ESPECTRO Laplaciana (Conectada: {topology_conn}) ===")
    print(ev_conn)
    print(f"componentes: {cc_conn}")
    print(f"lambda2 (conectividad algebraica): {lam2_conn:.6f}\n")

    print("=== ESPECTRO Laplaciana (Desconectada) ===")
    print(ev_disc)
    print(f"componentes: {cc_disc}")
    print(f"lambda2 (conectividad algebraica): {lam2_disc:.6f}\n")

    # Simulaciones
    t1, X1 = simulate_consensus_euler(L_conn, x0, T=6.0, dt=dt)
    plot_states(t1, X1, f"Consenso con red conectada ({topology_conn})", f"estados_conectada_{topology_conn}.png")

    t2, X2 = simulate_consensus_euler(L_disc, x0, T=6.0, dt=dt)
    plot_states(t2, X2, "Evolución con red desconectada (consenso por componente)", "estados_desconectada.png")

    # Switching (conectada -> desconectada -> conectada)
    segments = [(2.0, L_conn), (2.0, L_disc), (2.0, L_conn)]
    t3, X3 = simulate_switching_topology(segments, x0, dt=dt)
    plot_states(t3, X3, "Conectividad variante: conectada → desconectada → conectada", f"estados_switching_{topology_conn}.png")


def main():
    dt = 0.01

    # ====== ELIGE TOPOLOGÍA ======
    TOPOLOGY = "pentagram"  # "chain", "ring", "star", "complete", "pentagram" 
    RUN_ALL_TOPOLOGIES = False  

    # N recomendado por topología
    if TOPOLOGY == "pentagram":
        N = 5
    else:
        N = 8

    if RUN_ALL_TOPOLOGIES:
        for topo in ["chain", "ring", "star", "complete", "pentagram"]:
            n = 5 if topo == "pentagram" else 8
            run_case(N=n, dt=dt, topology_conn=topo)
    else:
        run_case(N=N, dt=dt, topology_conn=TOPOLOGY)


if __name__ == "__main__":
    main()
