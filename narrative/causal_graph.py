from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Set, Tuple


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


@dataclass(slots=True)
class CausalNode:
    node_id: str
    step: int
    role_name: str
    action_type: str = ""
    target_character: str = ""
    reason: str = ""
    text: str = ""
    scene_summary: str = ""
    environment_description: Dict[str, Any] = field(default_factory=dict)
    source_index: int = 0


class DirectedCausalGraph:
    """Directed causal graph built from SAM step_logs.

    This graph decouples fabula order (world progression) from syuzhet order
    (editing order). A node can be selected when all its predecessors are
    already selected.
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, CausalNode] = {}
        self.predecessors: Dict[str, Set[str]] = {}
        self.successors: Dict[str, Set[str]] = {}
        self._chronological_order: List[str] = []

    def add_node(self, node: CausalNode) -> None:
        if node.node_id in self.nodes:
            return
        self.nodes[node.node_id] = node
        self.predecessors[node.node_id] = set()
        self.successors[node.node_id] = set()
        self._chronological_order.append(node.node_id)

    def add_edge(self, src: str, dst: str) -> None:
        if src == dst:
            return
        if src not in self.nodes or dst not in self.nodes:
            return
        self.successors[src].add(dst)
        self.predecessors[dst].add(src)

    @classmethod
    def from_step_logs(
        cls,
        step_logs: List[Dict[str, Any]],
        *,
        add_role_chain: bool = True,
        add_target_dependency: bool = True,
        add_public_event_dependency: bool = True,
        max_public_event_deps_per_node: int = 3,
    ) -> "DirectedCausalGraph":
        graph = cls()

        rows: List[Tuple[int, int, Dict[str, Any], Dict[str, Any], str]] = []
        # (step, role_idx, step_item, role_item, node_id)
        for step_item in sorted(step_logs, key=lambda x: _safe_int(x.get("step", 0), 0)):
            step = _safe_int(step_item.get("step", 0), 0)
            if step <= 0:
                continue
            role_logs = step_item.get("role_logs", [])
            if not isinstance(role_logs, list):
                continue
            for ridx, role_item in enumerate(role_logs):
                if not isinstance(role_item, dict):
                    continue
                role_name = _safe_text(role_item.get("role_name"))
                if not role_name:
                    continue
                node_id = f"step::{step}::{role_name}"
                rows.append((step, ridx, step_item, role_item, node_id))

        rows.sort(key=lambda x: (x[0], x[1], x[4]))

        for src_idx, (step, _ridx, step_item, role_item, node_id) in enumerate(rows, start=1):
            next_action = role_item.get("next_action", {})
            if not isinstance(next_action, dict):
                next_action = {}
            node = CausalNode(
                node_id=node_id,
                step=step,
                role_name=_safe_text(role_item.get("role_name")),
                action_type=_safe_text(next_action.get("action_type")),
                target_character=_safe_text(next_action.get("target_character")),
                reason=_safe_text(next_action.get("reason")),
                text=_safe_text(role_item.get("role_output_text")),
                scene_summary=_safe_text(step_item.get("sam_summary")),
                environment_description=role_item.get("environment_description", {})
                if isinstance(role_item.get("environment_description", {}), dict)
                else {},
                source_index=src_idx,
            )
            graph.add_node(node)

        if not graph.nodes:
            return graph

        role_nodes: Dict[str, List[str]] = {}
        for nid in graph._chronological_order:
            role = graph.nodes[nid].role_name
            role_nodes.setdefault(role, []).append(nid)

        # 1) Same-role causal chain.
        if add_role_chain:
            for _role, ids in role_nodes.items():
                for i in range(1, len(ids)):
                    graph.add_edge(ids[i - 1], ids[i])

        # 2) Target-character dependency: link to latest target node before current.
        if add_target_dependency:
            role_positions: Dict[str, List[Tuple[int, str]]] = {}
            for nid in graph._chronological_order:
                n = graph.nodes[nid]
                role_positions.setdefault(n.role_name, []).append((n.source_index, nid))

            for nid in graph._chronological_order:
                node = graph.nodes[nid]
                target = node.target_character
                if not target or target not in role_positions:
                    continue
                latest_target_nid = ""
                for pos, tid in role_positions[target]:
                    if pos < node.source_index:
                        latest_target_nid = tid
                    else:
                        break
                if latest_target_nid:
                    graph.add_edge(latest_target_nid, nid)

        # 3) Environment dependency: consume observed public events as prerequisites.
        if add_public_event_dependency:
            event_to_latest_producer: Dict[str, str] = {}
            for nid in graph._chronological_order:
                node = graph.nodes[nid]
                env = node.environment_description if isinstance(node.environment_description, dict) else {}
                observed = env.get("public_events", [])
                consumed: List[str] = []
                if isinstance(observed, list):
                    for e in observed:
                        et = _safe_text(e)
                        if et and et in event_to_latest_producer:
                            consumed.append(event_to_latest_producer[et])
                if consumed:
                    uniq: List[str] = []
                    seen: Set[str] = set()
                    for dep in reversed(consumed):
                        if dep not in seen:
                            uniq.append(dep)
                            seen.add(dep)
                    for dep in uniq[: max(0, max_public_event_deps_per_node)]:
                        graph.add_edge(dep, nid)

                produced_text = _safe_text(node.text)
                if produced_text:
                    event_to_latest_producer[produced_text] = nid

        return graph

    def get_available_candidates(self, current_sequence: Iterable[str]) -> List[str]:
        selected: Set[str] = {str(x) for x in current_sequence if str(x) in self.nodes}
        available: List[str] = []
        for nid in self._chronological_order:
            if nid in selected:
                continue
            preds = self.predecessors.get(nid, set())
            if preds.issubset(selected):
                available.append(nid)
        return available

    def is_valid_append(self, current_sequence: Iterable[str], candidate_id: str) -> bool:
        if candidate_id not in self.nodes:
            return False
        selected: Set[str] = {str(x) for x in current_sequence if str(x) in self.nodes}
        if candidate_id in selected:
            return False
        preds = self.predecessors.get(candidate_id, set())
        return preds.issubset(selected)

    def remaining_blockers(self, current_sequence: Iterable[str], candidate_id: str) -> List[str]:
        if candidate_id not in self.nodes:
            return []
        selected: Set[str] = {str(x) for x in current_sequence if str(x) in self.nodes}
        preds = self.predecessors.get(candidate_id, set())
        return [nid for nid in self._chronological_order if nid in preds and nid not in selected]

    def topological_order(self) -> List[str]:
        # Kahn algorithm with chronological tie-break.
        indeg: Dict[str, int] = {nid: len(self.predecessors.get(nid, set())) for nid in self.nodes}
        order: List[str] = []
        queue: List[str] = [nid for nid in self._chronological_order if indeg[nid] == 0]
        head = 0
        while head < len(queue):
            nid = queue[head]
            head += 1
            order.append(nid)
            for nxt in sorted(self.successors.get(nid, set()), key=lambda x: self.nodes[x].source_index):
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    queue.append(nxt)
        if len(order) != len(self.nodes):
            # Cycle fallback.
            seen = set(order)
            for nid in self._chronological_order:
                if nid not in seen:
                    order.append(nid)
        return order

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_count": len(self.nodes),
            "edge_count": int(sum(len(v) for v in self.successors.values())),
            "nodes": [
                {
                    "node_id": n.node_id,
                    "step": n.step,
                    "role_name": n.role_name,
                    "action_type": n.action_type,
                    "target_character": n.target_character,
                    "reason": n.reason,
                    "text": n.text,
                    "scene_summary": n.scene_summary,
                    "source_index": n.source_index,
                    "predecessors": sorted(
                        self.predecessors.get(n.node_id, set()),
                        key=lambda x: self.nodes[x].source_index,
                    ),
                    "successors": sorted(
                        self.successors.get(n.node_id, set()),
                        key=lambda x: self.nodes[x].source_index,
                    ),
                }
                for n in (self.nodes[nid] for nid in self._chronological_order)
            ],
        }
