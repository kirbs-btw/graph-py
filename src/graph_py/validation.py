
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Literal, Optional

from pydantic import BaseModel, Field

from .core import Edge, Graph, Node, PropertyNode
from .ontology import Ontology, OntologyNode


class OntologyValidationIssue(BaseModel):
    """Single validation finding produced while comparing a graph to an ontology."""

    severity: Literal["error", "warning"]
    subject_type: Literal["graph", "node", "edge"]
    message: str
    subject_id: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class OntologyValidationReport(BaseModel):
    """Aggregated validation outcome."""

    issues: List[OntologyValidationIssue] = Field(default_factory=list)
    matched_nodes: Dict[str, str] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Return True when no errors were produced."""
        return not any(issue.severity == "error" for issue in self.issues)

    def add_issue(
        self,
        *,
        severity: Literal["error", "warning"],
        subject_type: Literal["graph", "node", "edge"],
        message: str,
        subject_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> OntologyValidationIssue:
        """Attach a validation issue to the report."""
        issue = OntologyValidationIssue(
            severity=severity,
            subject_type=subject_type,
            message=message,
            subject_id=subject_id,
            details=dict(details or {}),
        )
        self.issues.append(issue)
        return issue

    def extend(self, issues: Iterable[OntologyValidationIssue]) -> None:
        """Add multiple issues at once."""
        self.issues.extend(list(issues))


class OntologyValidator:
    """Validate arbitrary graphs against an ontology definition."""

    def __init__(self, ontology: Ontology):
        self.ontology = ontology

    def validate_graph(
        self,
        graph: Graph,
        *,
        strict: bool = True,
    ) -> OntologyValidationReport:
        """
        Validate a graph against this validator's ontology.

        When strict is True (default) missing concepts or relations are reported
        as errors; otherwise they degrade into warnings.
        """
        report = OntologyValidationReport()
        lookup_fail_severity: Literal["error", "warning"] = "error" if strict else "warning"
        matched_nodes: Dict[str, str] = {}

        for node in graph.nodes:
            concept = self._resolve_concept(node)
            if concept:
                matched_nodes[node.id] = concept.id
            else:
                report.add_issue(
                    severity=lookup_fail_severity,
                    subject_type="node",
                    subject_id=node.id,
                    message="Node cannot be mapped to any ontology concept.",
                    details=self._node_details(node),
                )

        for edge in graph.edges:
            source_concept = matched_nodes.get(edge.source)
            target_concept = matched_nodes.get(edge.target)
            if not source_concept or not target_concept:
                report.add_issue(
                    severity=lookup_fail_severity,
                    subject_type="edge",
                    subject_id=edge.id,
                    message="Edge connects nodes that are not mapped to ontology concepts.",
                    details={
                        "source": edge.source,
                        "target": edge.target,
                        "relation": self._relation_label(edge),
                    },
                )
                continue

            relation_label = self._relation_label(edge)
            if not relation_label:
                report.add_issue(
                    severity="warning",
                    subject_type="edge",
                    subject_id=edge.id,
                    message="Edge has no relation label; unable to match against ontology.",
                    details={
                        "source": source_concept,
                        "target": target_concept,
                    },
                )
                continue

            if not self._relation_exists(source_concept, target_concept, relation_label):
                report.add_issue(
                    severity=lookup_fail_severity,
                    subject_type="edge",
                    subject_id=edge.id,
                    message="Edge relation is not defined between mapped ontology concepts.",
                    details={
                        "source": source_concept,
                        "target": target_concept,
                        "relation": relation_label,
                    },
                )

        report.matched_nodes = matched_nodes
        return report

    def _resolve_concept(self, node: Node) -> Optional[OntologyNode]:
        """Find the ontology concept corresponding to the given node."""
        direct = self.ontology.get_concept(node.id)
        if direct:
            return direct

        for candidate in self._candidate_concept_identifiers(node):
            concept = self.ontology.get_concept(candidate)
            if concept:
                return concept
            concept = self.ontology.find_concept(candidate)
            if concept:
                return concept
        return None

    def _candidate_concept_identifiers(self, node: Node) -> List[str]:
        """Collect candidate identifiers (name, properties) for ontology lookup."""
        candidates: List[str] = []
        if node.name:
            candidates.append(node.name)
        if isinstance(node, PropertyNode):
            for key in ("concept_id", "concept", "ontology_id", "type"):
                value = node.properties.get(key)
                if isinstance(value, str) and value not in candidates:
                    candidates.append(value)
        return candidates

    def _node_details(self, node: Node) -> Dict[str, Any]:
        """Create a diagnostic snapshot for a graph node."""
        details: Dict[str, Any] = {}
        if node.name:
            details["name"] = node.name
        if isinstance(node, PropertyNode):
            details["properties"] = dict(node.properties)
        return details

    def _relation_label(self, edge: Edge) -> Optional[str]:
        """Return the relation label associated with the edge if available."""
        relation = getattr(edge, "relation", None)
        if relation:
            return relation
        return edge.name

    def _relation_exists(self, source_id: str, target_id: str, relation: str) -> bool:
        """Check whether an ontology relation exists between two concepts."""
        return any(
            edge.relation == relation
            for edge in self.ontology.iter_relations(source_id=source_id, target_id=target_id)
        )


__all__ = ["OntologyValidationIssue", "OntologyValidationReport", "OntologyValidator"]
