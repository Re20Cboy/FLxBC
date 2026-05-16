from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DIAGRAMS = (
    "docs/assets/flxbc-architecture-overview.svg",
    "docs/assets/flxbc-model-backends.svg",
    "docs/assets/flxbc-fl-strategies.svg",
)


def test_readme_references_architecture_diagrams() -> None:
    content = README.read_text(encoding="utf-8")

    for diagram in DIAGRAMS:
        assert diagram in content
        assert (ROOT / diagram).is_file()
        assert (ROOT / diagram.replace(".svg", ".png")).is_file()
