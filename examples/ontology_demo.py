"""Ontology validation example combining concept mapping and edge checks."""

from __future__ import annotations

from typing import Iterable

from graph_py import Edge, Graph
from graph_py.core import PropertyNode
from graph_py.ontology import Ontology, OntologyNode
from graph_py.validation import OntologyValidator


class ClinicalEdge(Edge):
    """Edge carrying relation metadata for ontology comparison."""

    relation: str


def build_clinical_ontology() -> Ontology:
    """
    Create a fragment of the ICD-10 based cardiometabolic ontology.

    Concept identifiers follow official ICD-10 codes for hypertension (I10),
    type 2 diabetes (E11), obesity (E66), and long-term insulin therapy (Z79.4).
    """
    ontology = Ontology(id="icd_fragment", name="Cardiometabolic Concepts")

    concepts: Iterable[OntologyNode] = (
        OntologyNode(
            id="I10",
            name="Essential (Primary) Hypertension",
            synonyms=["high blood pressure", "primary hypertension"],
            categories=["circulatory"],
        ),
        OntologyNode(
            id="E11",
            name="Type 2 Diabetes Mellitus",
            synonyms=["adult-onset diabetes", "non-insulin-dependent diabetes"],
            categories=["endocrine"],
        ),
        OntologyNode(
            id="E66",
            name="Obesity",
            synonyms=["BMI>=30"],
            categories=["endocrine"],
        ),
        OntologyNode(
            id="Z794",
            name="Long term (current) use of insulin",
            synonyms=["chronic insulin therapy"],
            categories=["therapy"],
        ),
    )
    for concept in concepts:
        ontology.add_node(concept)

    ontology.connect("E11", "I10", "comorbid_with")
    ontology.connect("E11", "Z794", "managed_by")
    ontology.connect("E11", "E66", "risk_factor")
    ontology.connect("I10", "E66", "risk_factor")

    return ontology


def build_patient_graph() -> Graph:
    """Represent a patient's condition graph referencing the ontology."""
    graph = Graph(id="patient_case_001", name="Clinic Intake Summary")

    findings: Iterable[PropertyNode] = (
        PropertyNode(
            id="n1",
            name="Hypertension",
            properties={"concept_id": "I10", "systolic_mmHg": 152},
        ),
        PropertyNode(
            id="n2",
            name="Type 2 Diabetes",
            properties={"concept_id": "E11", "hba1c": 8.3},
        ),
        PropertyNode(
            id="n3",
            name="Insulin Therapy",
            properties={"concept_id": "Z794"},
        ),
        PropertyNode(
            id="n4",
            name="Sleep Apnea",
            properties={"concept": "G4733"},
        ),
    )
    for finding in findings:
        graph.add_node(finding)

    relationships: Iterable[ClinicalEdge] = (
        ClinicalEdge(id="e1", source="n2", target="n1", relation="comorbid_with"),
        ClinicalEdge(id="e2", source="n2", target="n3", relation="managed_by"),
        ClinicalEdge(id="e3", source="n1", target="n4", relation="associated_with"),
    )
    for relation in relationships:
        graph.add_edge(relation)

    return graph


def main(strict: bool = False) -> None:
    ontology = build_clinical_ontology()
    patient_graph = build_patient_graph()
    validator = OntologyValidator(ontology)
    report = validator.validate_graph(patient_graph, strict=strict)

    print("Mapped nodes:")
    for node_id, concept_id in report.matched_nodes.items():
        concept = ontology.get_concept(concept_id)
        label = concept.label if concept else concept_id
        print(f"  {node_id} -> {concept_id} ({label})")

    print("\nIssues:")
    if not report.issues:
        print("  None")
    else:
        for issue in report.issues:
            details = ", ".join(f"{key}={value}" for key, value in issue.details.items())
            suffix = f" [{details}]" if details else ""
            print(f"  {issue.severity.upper()} {issue.subject_type} {issue.subject_id}: {issue.message}{suffix}")


if __name__ == "__main__":
    main()
