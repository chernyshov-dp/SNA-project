import pickle
import networkx as nx
import matplotlib.pyplot as plt


def create_company_graph(sp100_companies, company_features, sp100_index_price):
    """
    Create a graph representing SP100 companies and their sectors.

    Args:
    - sp100_companies (list): List of company symbols in SP100.
    - company_features (DataFrame): DataFrame with 'Symbol' and 'Sector' columns.
    - sp100_index_price (float): Current SP100 index price.

    Returns:
    - nx.Graph: NetworkX graph object representing company-sector relationships.
    """
    G = nx.Graph()
    G.add_node("SP100", index_price=sp100_index_price)

    sector_nodes = {}
    
    # Add companies as nodes and connect them to sector nodes
    for _, row in company_features.iterrows():
        symbol = row["Symbol"]
        sector = row["Sector"]
        
        G.add_node(symbol, sector=sector)
        
        # Create a sector node if it does not already exist
        if sector not in sector_nodes:
            sector_nodes[sector] = f'Sector_{sector}'
            G.add_node(sector_nodes[sector])
        
        # Connecting the company to the sector node
        G.add_edge(sector_nodes[sector], symbol)
    
    # Connect all sector nodes to the 'SP100' node
    for sector_node in sector_nodes.values():
        G.add_edge("SP100", sector_node)
    
    return G


def draw_company_graph(G, legend=False, save=False, filename="graph.png"):
    """
    Draw a network graph of companies and their sectors.

    Args:
    - G (nx.Graph): NetworkX graph object to draw.
    - legend (bool, optional): Whether to include a sector color legend. Default is False.
    - save (bool, optional): Whether to save the plot. Default is False.
    - filename (str, optional): Filename for saved plot. Default is "graph.png".

    Returns:
    - None
    """
    if save:
        sector_colors = {
            "Information Technology": "blue",
            "Health Care": "green",
            "Consumer Discretionary": "red",
            "Financials": "purple",
            "Consumer Staples": "orange",
            "Utilities": "brown",
            "Real Estate": "pink",
            "Materials": "yellow",
            "Industrials": "cyan",
            "Energy": "gray",
            "Communication Services": "magenta"
        }
        
        node_colors = []
        for node in G.nodes:
            if node.startswith("Sector_"):
                sector = node.split('_')[1]
                node_colors.append(sector_colors.get(sector, "lightblue"))
            else:
                node_colors.append("lightblue")
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=200)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
        
        legend_handles = [plt.Line2D([0],[0], marker="o",
                                     color=color, label=label, markersize=10,
                                     linestyle="None") for label, color in sector_colors.items()]

        if legend:
            plt.legend(handles=legend_handles)

        plt.axis("off")
        plt.savefig(filename)
        print(f" - Successfully saves as {filename}")


def main():
    files_to_load = [
        "sp100_companies.pkl",
        "sp100_index_price.pkl",
        "company_features.pkl",
    ]

    loaded_data = {}

    for file_name in files_to_load:
        with open(file_name, "rb") as f:
            loaded_data[file_name[:-4]] = pickle.load(f)

    sp100_companies = loaded_data["sp100_companies"]
    sp100_index_price = loaded_data["sp100_index_price"]
    company_features = loaded_data["company_features"]

    company_graph = create_company_graph(sp100_companies, company_features, sp100_index_price)
    draw_company_graph(company_graph, save=True, filename="graph.png")


if __name__ == "__main__":
    main()
