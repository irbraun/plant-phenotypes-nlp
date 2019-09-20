
# How to merge the graphs without problems with undirected/directed order of nodes stuff.
# (Things that make doing a normal outer-join not really appropriate here).
import networkx as nx
df1 = get_edgelist_with_bagofwords(descriptions)
df2 = get_edgelist_with_setofwords(descriptions)

df1["bow_value"] = df1["value"]
df2["sow_value"] = df2["value"]

g1 = nx.from_pandas_edgelist(df1, source="from", target="to", edge_attr=["bow_value"]).to_undirected()
g2 = nx.from_pandas_edgelist(df2, source="from", target="to", edge_attr=["sow_value"]).to_undirected()

mgraph = nx.MultiGraph()

# From https://networkx.github.io/documentation/networkx-1.9.1/reference/classes.multigraph.html:
# "If some edges connect nodes not yet in the graph, the nodes are added automatically. If an edge 
# already exists, an additional edge is created and stored using a key to identify the edge. By 
# default the key is the lowest unused integer.""

mgraph.add_edges_from(g1.edges())
mgraph.add_edges_from(g2.edges())

l = nx.to_pandas_edgelist(mgraph, source="from", target="to")

print(l)

sys.exit()



#v = g1.get_edge_data(1,2)["value"]
#print(v)
#xys.exit()









g_df1 = nx.from_pandas_edgelist(df1, source="from", target="to", edge_attr=["value"]).to_undirected()


for (node1,node2,data) in g_df1.edges(data=True):
	print(node1, node2, data)
    #all_weights.append(data['weight'])

sys.exit()


g_df2 = nx.from_pandas_edgelist(df2, source="from", target="to", edge_attr=["value"]).to_undirected()
mgraph = nx.MultiGraph()

# From https://networkx.github.io/documentation/networkx-1.9.1/reference/classes.multigraph.html:
# "If some edges connect nodes not yet in the graph, the nodes are added automatically. If an edge 
# already exists, an additional edge is created and stored using a key to identify the edge. By 
# default the key is the lowest unused integer.""

print(g_df1.edges())


mgraph.add_weighted_edges_from(g_df1.edges())
mgraph.add_weighted_edges_from(g_df2.edges())
mgraph = mgraph.to_undirected()
print(mgraph.edges())

l = nx.to_pandas_edgelist(mgraph, source="from", target="to")

print(l)
print(l.dtypes)