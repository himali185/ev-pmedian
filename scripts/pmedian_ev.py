#!/usr/bin/env python3
"""
pmedian_ev.py (LOW-RAM / LOW-DISK MODE) - FIXED

- Handles OSMnx variants that lack geometries_from_point
- Uses node index directly to pick candidate node IDs (robust)
- Minimal outputs, designed to run within ~1 GB
"""
import os, time, json
import numpy as np
import osmnx as ox
import geopandas as gpd
import networkx as nx
import pandas as pd
import folium
import pulp
from shapely.geometry import Point

# -------------------- TUNABLES (LOW-RESOURCE) --------------------
USE_BBOX = False                   # use graph_from_point (fast)
CENTER_POINT = (13.02, 77.59)      # lat,lon center
POINT_RADIUS_M = 1000              # 1 km radius -> much smaller graph
NETWORK_TYPE = "drive"
DEMAND_SAMPLE = 30                 # tiny demand sample
CANDIDATE_MAX = 20                 # tiny candidate set
P_FACILITIES = 1                   # choose 1 site to keep ILP small
OUTDIR = os.path.join(os.getcwd(), "outputs")
# ---------------------------------------------------------------

# Disable OSMnx disk cache to avoid large files
ox.settings.use_cache = False
ox.settings.log_console = False

def ensure_outdir():
    os.makedirs(OUTDIR, exist_ok=True)

def get_graph():
    lat, lon = CENTER_POINT
    print(f"Downloading graph_from_point at {CENTER_POINT} radius {POINT_RADIUS_M} m ...")
    # Some osmnx variants accept graph_from_point((lat,lon), dist=...), others differ.
    # This call works for the typical versions installed via pip.
    try:
        G = ox.graph_from_point((lat, lon), dist=POINT_RADIUS_M, network_type=NETWORK_TYPE)
    except TypeError:
        # fallback: compute a small bbox approx and use graph_from_bbox((north,south,east,west))
        deg = POINT_RADIUS_M / 111000.0
        north = lat + deg
        south = lat - deg
        east = lon + deg
        west = lon - deg
        G = ox.graph_from_bbox((north, south, east, west), network_type=NETWORK_TYPE)
    return G

def build_demand_points(nodes_gdf, buildings_gdf=None, n_sample=DEMAND_SAMPLE):
    """Prefer building centroids if available, otherwise sample nodes."""
    if buildings_gdf is not None and (not buildings_gdf.empty):
        centroids = buildings_gdf.geometry.centroid
        g = gpd.GeoDataFrame(geometry=centroids, crs=buildings_gdf.crs)
        if len(g) > n_sample:
            g = g.sample(n_sample, random_state=1)
        g = g.reset_index(drop=True)
        g["demand"] = 1.0
        g = g.to_crs(nodes_gdf.crs)
        return g
    else:
        sampled = nodes_gdf.sample(n_sample, random_state=2).reset_index()
        # sampled now has a column 'index' with node ids and a geometry column
        # Keep geometry only and set demand
        sampled = sampled[['geometry']].copy()
        sampled["demand"] = 1.0
        sampled = sampled.set_crs(nodes_gdf.crs)
        return sampled

def compute_cost_matrix(G_proj, demand_gdf, candidate_nodes_list):
    """Compute network distances demand->candidate for small matrices."""
    demand_node_ids = [ox.nearest_nodes(G_proj, geom.x, geom.y) for geom in demand_gdf.geometry]
    n_dem = len(demand_node_ids)
    m_cand = len(candidate_nodes_list)
    cost = np.full((n_dem, m_cand), fill_value=1e9, dtype=float)

    node_to_demand_idx = {}
    for i, nid in enumerate(demand_node_ids):
        node_to_demand_idx.setdefault(nid, []).append(i)

    for j, cand_nid in enumerate(candidate_nodes_list):
        lengths = nx.single_source_dijkstra_path_length(G_proj, cand_nid, weight='length')
        for demand_nid, idx_list in node_to_demand_idx.items():
            d = lengths.get(demand_nid, 1e9)
            for idx in idx_list:
                cost[idx, j] = d

    col_names = [f"cand_{i}" for i in range(m_cand)]
    cost_df = pd.DataFrame(cost, index=range(n_dem), columns=col_names)
    return cost_df, demand_node_ids

def solve_p_median(cost_df, demand_weights, p=P_FACILITIES, time_limit=20):
    n_dem, m_cand = cost_df.shape
    model = pulp.LpProblem("p_median", pulp.LpMinimize)
    x = pulp.LpVariable.dicts('x', (range(n_dem), range(m_cand)), cat='Binary')
    y = pulp.LpVariable.dicts('y', range(m_cand), cat='Binary')
    model += pulp.lpSum([demand_weights[i] * cost_df.iat[i, j] * x[i][j]
                         for i in range(n_dem) for j in range(m_cand)])
    for i in range(n_dem):
        model += pulp.lpSum([x[i][j] for j in range(m_cand)]) == 1
    for i in range(n_dem):
        for j in range(m_cand):
            model += x[i][j] <= y[j]
    model += pulp.lpSum([y[j] for j in range(m_cand)]) == p
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    model.solve(solver)
    chosen = [j for j in range(m_cand) if pulp.value(y[j]) > 0.5]
    return chosen

if __name__ == "__main__":
    t0 = time.time()
    ensure_outdir()

    print("1) Downloading small road network ...")
    G = get_graph()
    print("Nodes:", len(G.nodes), "Edges:", len(G.edges))

    print("2) Projecting graph ...")
    G_proj = ox.project_graph(G)
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_proj, nodes=True, edges=True)

    # 3) Try building footprints (handle osmnx variants gracefully)
    print("3) Attempting building footprints (may not be available)...")
    buildings = None
    try:
        lat, lon = CENTER_POINT
        try:
            buildings = ox.geometries_from_point((lat, lon), tags={"building": True}, dist=POINT_RADIUS_M)
        except Exception:
            # fallback to bbox approx
            deg = POINT_RADIUS_M / 111000.0
            north = lat + deg; south = lat - deg; east = lon + deg; west = lon - deg
            try:
                buildings = ox.geometries_from_bbox((north, south, east, west), tags={"building": True})
            except Exception:
                buildings = None
        if buildings is not None and not buildings.empty:
            buildings = buildings[buildings.geometry.notnull()].copy().to_crs(nodes_gdf.crs)
            print("  buildings:", len(buildings))
        else:
            print("  no building footprints available (continuing).")
            buildings = None
    except Exception as e:
        print("  building fetch error (continuing):", e)
        buildings = None

    # 4) demand points
    print("4) Building demand points (sampling)...")
    demand_gdf = build_demand_points(nodes_gdf, buildings_gdf=buildings, n_sample=DEMAND_SAMPLE)
    demand_gdf = demand_gdf.to_crs(nodes_gdf.crs).reset_index(drop=True)

    # 5) existing chargers (radial / bbox fallback)
    print("5) Looking for existing charging_station POIs (may be none)...")
    chargers = None
    try:
        lat, lon = CENTER_POINT
        try:
            chargers = ox.geometries_from_point((lat, lon), tags={"amenity": "charging_station"}, dist=POINT_RADIUS_M)
        except Exception:
            deg = POINT_RADIUS_M / 111000.0
            north = lat + deg; south = lat - deg; east = lon + deg; west = lon - deg
            try:
                chargers = ox.geometries_from_bbox((north, south, east, west), tags={"amenity": "charging_station"})
            except Exception:
                chargers = None
        if chargers is not None and not chargers.empty:
            chargers = chargers[chargers.geometry.notnull()].copy().to_crs(nodes_gdf.crs)
            print("  chargers:", len(chargers))
        else:
            chargers = None
            print("  no chargers in OSM for this small area.")
    except Exception:
        chargers = None

    # 6) candidate selection (robust: use index for node IDs)
    print("6) Selecting candidate nodes (top junctions)...")
    nodes_gdf['degree'] = [G_proj.degree[nid] for nid in nodes_gdf.index]
    charger_nodes = []
    if chargers is not None:
        # snap chargers to nearest nodes (use explicit params to be robust)
        charger_nodes = [ox.nearest_nodes(G_proj, X=float(geom.x), Y=float(geom.y)) for geom in chargers.geometry]

    # pick top-degree nodes excluding charger nodes; **do not reset index for node IDs**
    candidate_pool = nodes_gdf[~nodes_gdf.index.isin(charger_nodes)].copy()
    candidate_pool = candidate_pool.sort_values('degree', ascending=False).head(CANDIDATE_MAX)
    candidate_nodes = list(candidate_pool.index)  # <-- robust extraction of node IDs
    print("  selected candidate nodes:", len(candidate_nodes))

    # 7) compute cost matrix
    print("7) Computing cost matrix (small)...")
    cost_df, demand_node_ids = compute_cost_matrix(G_proj, demand_gdf, candidate_nodes)
    demand_weights = np.array(demand_gdf['demand'].values, dtype=float)

    # 8) solve p-median
    print(f"8) Solving p-median for p={P_FACILITIES} ...")
    chosen_idx = solve_p_median(cost_df, demand_weights, p=P_FACILITIES, time_limit=20)
    chosen_nodes = [candidate_nodes[i] for i in chosen_idx] if chosen_idx else []
    print("  chosen node IDs:", chosen_nodes)

    # 9) minimal outputs (convert chosen to latlon)
    chosen_geoms = [nodes_gdf.loc[nid].geometry for nid in chosen_nodes] if chosen_nodes else []
    chosen_gdf = gpd.GeoDataFrame(geometry=chosen_geoms, crs=nodes_gdf.crs).to_crs(epsg=4326) if chosen_geoms else gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    print("10) Saving outputs (minimal)...")
    demand_gdf.to_crs(epsg=4326).to_file(os.path.join(OUTDIR, "demand_sample.geojson"), driver="GeoJSON")
    # for candidates.csv we will export node id as well
    cand_export = candidate_pool.reset_index().rename(columns={'index':'node_id'})
    cand_export.to_crs(epsg=4326).to_file(os.path.join(OUTDIR, "candidates.geojson"), driver="GeoJSON")
    chosen_gdf.to_file(os.path.join(OUTDIR, "chosen_sites.geojson"), driver="GeoJSON")
    if chargers is not None:
        chargers.to_crs(epsg=4326).to_file(os.path.join(OUTDIR, "existing_chargers.geojson"), driver="GeoJSON")

    # 11) metrics: before/after avg network distance
    und = G_proj.to_undirected()
    if charger_nodes:
        lengths_before = nx.multi_source_dijkstra_path_length(und, charger_nodes, weight='length')
        dist_before = np.array([lengths_before.get(dnid, 1e9) for dnid in demand_node_ids], dtype=float)
    else:
        dist_before = np.array([1e9]*len(demand_node_ids), dtype=float)
    combined = charger_nodes + chosen_nodes
    lengths_after = nx.multi_source_dijkstra_path_length(und, combined, weight='length') if combined else {}
    dist_after = np.array([lengths_after.get(dnid, 1e9) for dnid in demand_node_ids], dtype=float) if combined else dist_before
    finite_before = dist_before[dist_before < 1e8]
    finite_after = dist_after[dist_after < 1e8]
    avg_before = float(np.mean(finite_before)) if finite_before.size>0 else None
    avg_after = float(np.mean(finite_after)) if finite_after.size>0 else None
    reduction_pct = (100.0*(avg_before-avg_after)/avg_before) if (avg_before and avg_after) else None
    metrics = {"avg_before_m": avg_before, "avg_after_m": avg_after, "reduction_pct": reduction_pct}
    with open(os.path.join(OUTDIR, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("Metrics:", metrics)

    # 12) interactive map
    demand_latlon = demand_gdf.to_crs(epsg=4326)
    if not demand_latlon.empty:
        center = [demand_latlon.geometry.unary_union.centroid.y, demand_latlon.geometry.unary_union.centroid.x]
    else:
        center = [CENTER_POINT[0], CENTER_POINT[1]]
    m = folium.Map(location=center, zoom_start=14, tiles="cartodbpositron")
    for _, r in demand_latlon.iterrows():
        folium.CircleMarker([r.geometry.y, r.geometry.x], radius=2, color='grey', fill=True, fillOpacity=0.6).add_to(m)
    for _, r in cand_export.to_crs(epsg=4326).iterrows():
        folium.CircleMarker([r.geometry.y, r.geometry.x], radius=3, color='blue', fill=True, fillOpacity=0.7).add_to(m)
    for _, r in chosen_gdf.iterrows():
        folium.CircleMarker([r.geometry.y, r.geometry.x], radius=6, color='red', fill=True, fillOpacity=0.9).add_to(m)
    if chargers is not None:
        for _, r in chargers.to_crs(epsg=4326).iterrows():
            folium.CircleMarker([r.geometry.y, r.geometry.x], radius=4, color='darkred', fill=True, fillOpacity=0.9).add_to(m)
    map_path = os.path.join(OUTDIR, "pmedian_map.html")
    m.save(map_path)

    print("Done. Outputs in:", OUTDIR)
    print("Open:", map_path)
    print("Elapsed (s):", time.time() - t0)

