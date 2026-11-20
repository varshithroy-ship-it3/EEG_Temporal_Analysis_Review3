"""
NetworkX Relationship Graph Generator.
Builds an evidence-linked directed graph of temporal events and interval relationships,
rendering publication-grade diagrams and supporting interactive visualization.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


class GraphGenerator:
    """
    Constructs and visualizes directed graphs of temporal interval relationships.
    """

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_graph(
        self,
        events_df: pd.DataFrame,
        rel_df: pd.DataFrame,
    ) -> nx.DiGraph:
        """
        Constructs a NetworkX DiGraph from event and relationship tables.
        """
        G = nx.DiGraph()

        # Add Nodes
        for _, row in events_df.iterrows():
            evt_id = str(row["event_id"])
            rec_id = str(row["recording_id"])
            source = str(row["source"])
            start_s = float(row["start_seconds"])
            end_s = float(row["end_seconds"])
            dur = end_s - start_s
            conf = row.get("confidence")

            is_expert = source == "expert_annotation"
            node_type = "Expert" if is_expert else "Model Pred"

            conf_label = f"\nConf: {conf*100:.0f}%" if conf is not None and not is_expert else ""
            display_label = f"{evt_id}\n[{start_s:.1f}s - {end_s:.1f}s]\n({dur:.1f}s){conf_label}"

            G.add_node(
                evt_id,
                label=display_label,
                recording_id=rec_id,
                source=source,
                is_expert=is_expert,
                start_seconds=start_s,
                end_seconds=end_s,
                duration=dur,
                confidence=conf,
            )

        # Add Edges
        for _, row in rel_df.iterrows():
            u = str(row["event_a_id"])
            v = str(row["event_b_id"])
            rel_type = str(row["relation_type"])
            overlap = float(row.get("overlap_seconds", 0.0))
            gap = float(row.get("gap_seconds", 0.0))

            if overlap > 0:
                edge_label = f"{rel_type}\n({overlap:.1f}s overlap)"
            elif gap > 0:
                edge_label = f"{rel_type}\n({gap:.1f}s gap)"
            else:
                edge_label = rel_type

            if u in G and v in G:
                G.add_edge(
                    u, v,
                    relation_type=rel_type,
                    overlap_seconds=overlap,
                    gap_seconds=gap,
                    label=edge_label,
                    rel_id=str(row["relationship_id"]),
                )

        return G

    def render_and_save(
        self,
        G: nx.DiGraph,
        filename: str = "graph.png",
    ) -> str:
        """
        Renders the graph into a polished high-resolution image with distinct styling
        for model predictions and expert annotations.
        """
        out_path = self.output_dir / filename
        if len(G.nodes) == 0:
            # Empty graph placeholder
            fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
            fig.patch.set_facecolor("#0F172A")
            ax.set_facecolor("#0F172A")
            ax.text(
                0.5, 0.5, "No temporal relationships detected to graph.",
                color="#94A3B8", ha="center", va="center", fontsize=14
            )
            plt.savefig(out_path, facecolor=fig.get_facecolor())
            plt.close(fig)
            return str(out_path)

        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
        fig.patch.set_facecolor("#0F172A")
        ax.set_facecolor("#0F172A")

        # Partition nodes by recording and source
        pos = nx.spring_layout(G, k=1.8, iterations=60, seed=42)

        expert_nodes = [n for n, attr in G.nodes(data=True) if attr.get("is_expert", False)]
        pred_nodes = [n for n, attr in G.nodes(data=True) if not attr.get("is_expert", False)]

        # Draw Expert Nodes (Teal / Emerald, Rectangular feel)
        if expert_nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=expert_nodes,
                node_color="#0D9488",  # Teal 600
                node_size=3200,
                node_shape="s",
                edgecolors="#2DD4BF",  # Teal 400
                linewidths=2.5,
                ax=ax,
            )

        # Draw Model Predicted Nodes (Purple / Violet, Circular)
        if pred_nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=pred_nodes,
                node_color="#7C3AED",  # Violet 600
                node_size=2800,
                node_shape="o",
                edgecolors="#C084FC",  # Violet 400
                linewidths=2.5,
                ax=ax,
            )

        # Draw Directed Edges
        nx.draw_networkx_edges(
            G, pos,
            edge_color="#38BDF8",  # Sky 400
            arrows=True,
            arrowsize=22,
            arrowstyle="-|>",
            width=2.0,
            connectionstyle="arc3,rad=0.15",
            ax=ax,
        )

        # Draw Node Labels
        node_labels = {n: attr.get("label", n) for n, attr in G.nodes(data=True)}
        nx.draw_networkx_labels(
            G, pos,
            labels=node_labels,
            font_size=8.5,
            font_weight="bold",
            font_color="#F8FAFC",
            ax=ax,
        )

        # Draw Edge Labels
        edge_labels = {(u, v): d.get("label", "") for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_labels,
            font_size=8,
            font_color="#FCD34D",  # Amber 300
            bbox=dict(boxstyle="round,pad=0.2", fc="#1E293B", ec="#475569", alpha=0.9),
            ax=ax,
        )

        ax.set_title(
            "EEG Temporal Progression & Interval Relationship Graph\n"
            "Teal Square = Expert Annotation | Violet Circle = Model Prediction",
            fontsize=13, weight="bold", color="#F8FAFC", pad=16
        )
        ax.axis("off")

        plt.tight_layout()
        plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

        return str(out_path)
