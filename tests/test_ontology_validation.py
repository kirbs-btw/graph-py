from graph_py import Edge, Graph
from graph_py.core import PropertyNode
from graph_py.ontology import Ontology, OntologyNode
from graph_py.validation import OntologyValidator


class RelationEdge(Edge):
    relation: str


def test_validator_matches_synonym_casefold():
    ontology = Ontology(id="icd")
    ontology.add_node(
        OntologyNode(
            id="E11",
            name="Type 2 Diabetes Mellitus",
            synonyms=["type 2 diabetes", "adult-onset diabetes"],
        )
    )

    graph = Graph(id="patient", name="Case Study")
    graph.add_node(PropertyNode(id="n1", name="TYPE 2 DIABETES"))

    validator = OntologyValidator(ontology)
    report = validator.validate_graph(graph)

    assert report.is_valid
    assert report.matched_nodes["n1"] == "E11"


def test_validator_reports_unknown_relation():
    ontology = Ontology(id="icd")
    ontology.add_node(OntologyNode(id="E11", name="Type 2 Diabetes"))
    ontology.add_node(OntologyNode(id="I10", name="Essential Hypertension"))
    ontology.connect("E11", "I10", "comorbid_with")

    graph = Graph(id="patient", name="Case Study")
    graph.add_node(PropertyNode(id="n1", properties={"concept_id": "E11"}))
    graph.add_node(PropertyNode(id="n2", properties={"concept_id": "I10"}))
    graph.add_edge(RelationEdge(id="e1", source="n1", target="n2", relation="contraindicated"))

    validator = OntologyValidator(ontology)
    report = validator.validate_graph(graph)

    assert len(report.issues) == 1
    issue = report.issues[0]
    assert issue.severity == "error"
    assert issue.subject_type == "edge"
    assert issue.message.startswith("Edge relation is not defined")
    assert issue.details["relation"] == "contraindicated"
