from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, List, Optional

from pydantic import Field, PrivateAttr

from .core import Edge, Graph, Node, PropertyNode


def _deduplicate_preserve_order(values: Iterable[str]) -> List[str]:
    """Return values without duplicates while keeping the original order."""
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        cleaned = value.strip()
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


class OntologyNode(PropertyNode):
    """Node representing a concept in an ontology."""

    description: Optional[str] = None
    synonyms: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)

    def add_synonym(self, synonym: str) -> bool:
        """Attach a synonym to this concept; return True when it was new."""
        cleaned = synonym.strip()
        if not cleaned:
            return False
        key = cleaned.casefold()
        existing = {entry.casefold() for entry in self.synonyms}
        if key in existing:
            return False
        self.synonyms.append(cleaned)
        return True

    def remove_synonym(self, synonym: str) -> bool:
        """Remove a synonym if present; returns True when removed."""
        for index, entry in enumerate(self.synonyms):
            if entry.casefold() == synonym.strip().casefold():
                del self.synonyms[index]
                return True
        return False

    @property
    def label(self) -> str:
        """Primary label for the concept."""
        return self.name or self.id


class OntologyEdge(Edge):
    """Edge describing a semantic relation between ontology nodes."""

    relation: str
    weight: Optional[float] = None
    qualifiers: Dict[str, Any] = Field(default_factory=dict)


class Ontology(Graph):
    """Graph specialization that keeps track of ontology concepts and relations."""

    _label_index: Dict[str, str] = PrivateAttr(default_factory=dict)
    _synonym_index: Dict[str, str] = PrivateAttr(default_factory=dict)

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._rebuild_indexes()

    # ------------------------------------------------------------------ #
    # Concept management
    # ------------------------------------------------------------------ #

    def add_node(self, node: Node) -> None:
        """Add an ontology concept; only OntologyNode instances are accepted."""
        if not isinstance(node, OntologyNode):
            raise TypeError("Ontology.add_node expects an OntologyNode instance")
        node.synonyms = _deduplicate_preserve_order(node.synonyms)
        node.categories = _deduplicate_preserve_order(node.categories)
        super().add_node(node)
        self._register_concept(node)

    def add_concept(
        self,
        concept_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        synonyms: Optional[Iterable[str]] = None,
        categories: Optional[Iterable[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> OntologyNode:
        """Create a concept on the fly and add it to the ontology."""
        node = OntologyNode(
            id=concept_id,
            name=name,
            description=description,
            synonyms=_deduplicate_preserve_order(synonyms or []),
            categories=_deduplicate_preserve_order(categories or []),
            properties=dict(properties or {}),
        )
        self.add_node(node)
        return node

    def get_concept(self, concept_id: str) -> Optional[OntologyNode]:
        """Return the concept with the given identifier if present."""
        node = super().get_node(concept_id)
        return node if isinstance(node, OntologyNode) else None

    def find_concept(self, term: str, *, case_sensitive: bool = False) -> Optional[OntologyNode]:
        """
        Look up a concept by label or synonym.

        When case_sensitive is False (the default), lookups use a casefolded
        index; otherwise a linear scan is performed to match the exact casing.
        """
        if not term:
            return None

        if not case_sensitive:
            key = self._normalise(term)
            concept_id = self._synonym_index.get(key) or self._label_index.get(key)
            if not concept_id:
                return None
            return self.get_concept(concept_id)

        # Case-sensitive lookup falls back to an explicit scan.
        for node in self.iter_concepts():
            if node.label == term or any(syn == term for syn in node.synonyms):
                return node
        return None

    def add_synonym(self, concept_id: str, synonym: str) -> None:
        """Attach a synonym to a concept and update the index."""
        node = self._require_concept(concept_id)
        cleaned = synonym.strip()
        if not cleaned:
            raise ValueError("synonym must not be empty")
        key = self._normalise(cleaned)
        existing = self._synonym_index.get(key)
        if existing and existing != concept_id:
            raise ValueError(f"synonym '{synonym}' already assigned to '{existing}'")
        if node.add_synonym(cleaned):
            self._synonym_index[key] = concept_id

    def remove_synonym(self, concept_id: str, synonym: str) -> bool:
        """Remove a synonym and update the index; returns True on removal."""
        node = self._require_concept(concept_id)
        removed = node.remove_synonym(synonym)
        if removed:
            key = self._normalise(synonym)
            current = self._synonym_index.get(key)
            if current == concept_id:
                del self._synonym_index[key]
        return removed

    def synonyms_for(self, concept_id: str) -> List[str]:
        """Return all synonyms stored for a concept."""
        node = self._require_concept(concept_id)
        return list(node.synonyms)

    def iter_concepts(self) -> Iterator[OntologyNode]:
        """Yield all ontology concepts."""
        for node in self.nodes:
            if isinstance(node, OntologyNode):
                yield node

    # ------------------------------------------------------------------ #
    # Relation management
    # ------------------------------------------------------------------ #

    def add_edge(self, edge: Edge) -> None:
        """Add a semantic relation between two concepts."""
        if not isinstance(edge, OntologyEdge):
            raise TypeError("Ontology.add_edge expects an OntologyEdge instance")
        if not self.get_concept(edge.source) or not self.get_concept(edge.target):
            raise ValueError("Ontology edges can only connect concepts already in the ontology")
        if not edge.name:
            edge.name = edge.relation
        super().add_edge(edge)

    def add_relation(self, relation: OntologyEdge) -> OntologyEdge:
        """Wrapper around add_edge returning the relation."""
        self.add_edge(relation)
        return relation

    def connect(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        *,
        relation_id: Optional[str] = None,
        weight: Optional[float] = None,
        qualifiers: Optional[Dict[str, Any]] = None,
        allow_parallel: bool = False,
    ) -> OntologyEdge:
        """
        Create an ontology relation between two concepts.

        By default, an identical relation (same source, target, relation label)
        cannot be added twice unless allow_parallel=True.
        """
        self._require_concept(source_id)
        self._require_concept(target_id)
        if not relation:
            raise ValueError("relation must not be empty")

        if not allow_parallel and self._relation_exists(source_id, target_id, relation):
            raise ValueError(
                f"relation '{relation}' between '{source_id}' and '{target_id}' already exists"
            )

        edge_id = relation_id or self._build_edge_id(source_id, relation, target_id)
        edge = OntologyEdge(
            id=edge_id,
            name=relation,
            relation=relation,
            source=source_id,
            target=target_id,
            weight=weight,
            qualifiers=dict(qualifiers or {}),
        )
        self.add_edge(edge)
        return edge

    def iter_relations(
        self,
        *,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        relation: Optional[str] = None,
    ) -> Iterator[OntologyEdge]:
        """Yield relations filtered by optional attributes."""
        for edge in self.edges:
            if not isinstance(edge, OntologyEdge):
                continue
            if source_id and edge.source != source_id:
                continue
            if target_id and edge.target != target_id:
                continue
            if relation and edge.relation != relation:
                continue
            yield edge

    def parents(self, concept_id: str, relation: Optional[str] = None) -> List[OntologyNode]:
        """Return all parent concepts linked by incoming relations."""
        self._require_concept(concept_id)
        parents: List[OntologyNode] = []
        for edge in self.iter_relations(target_id=concept_id, relation=relation):
            parent = self.get_concept(edge.source)
            if parent:
                parents.append(parent)
        return parents

    def children(self, concept_id: str, relation: Optional[str] = None) -> List[OntologyNode]:
        """Return all child concepts linked by outgoing relations."""
        self._require_concept(concept_id)
        children: List[OntologyNode] = []
        for edge in self.iter_relations(source_id=concept_id, relation=relation):
            child = self.get_concept(edge.target)
            if child:
                children.append(child)
        return children

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _rebuild_indexes(self) -> None:
        """Recreate fast lookup indexes from the current node set."""
        self._label_index.clear()
        self._synonym_index.clear()
        for node in self.iter_concepts():
            self._register_concept(node)

    def _register_concept(self, node: OntologyNode) -> None:
        """Register a concept and its synonyms in the lookup indexes."""
        label = node.label.strip()
        if not label:
            raise ValueError("ontology concepts require a non-empty name or id")
        self._register_label(label, node.id)
        for synonym in node.synonyms:
            self._register_synonym(synonym, node.id)

    def _register_label(self, label: str, concept_id: str) -> None:
        key = self._normalise(label)
        existing = self._label_index.get(key)
        if existing and existing != concept_id:
            raise ValueError(f"label '{label}' already assigned to '{existing}'")
        self._label_index[key] = concept_id
        # Also allow lookups through the label as a synonym.
        self._synonym_index[key] = concept_id

    def _register_synonym(self, synonym: str, concept_id: str) -> None:
        key = self._normalise(synonym)
        existing = self._synonym_index.get(key)
        if existing and existing != concept_id:
            raise ValueError(f"synonym '{synonym}' already assigned to '{existing}'")
        self._synonym_index[key] = concept_id

    def _relation_exists(self, source_id: str, target_id: str, relation: str) -> bool:
        return any(
            edge for edge in self.iter_relations(source_id=source_id, target_id=target_id)
            if edge.relation == relation
        )

    def _require_concept(self, concept_id: str) -> OntologyNode:
        node = self.get_concept(concept_id)
        if not node:
            raise KeyError(f"concept '{concept_id}' is not part of the ontology")
        return node

    @staticmethod
    def _normalise(term: str) -> str:
        return term.casefold()

    @staticmethod
    def _build_edge_id(source_id: str, relation: str, target_id: str) -> str:
        base = f"{source_id}:{relation}:{target_id}"
        return base


__all__ = ["OntologyNode", "OntologyEdge", "Ontology"]

