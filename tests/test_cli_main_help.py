from click.testing import CliRunner

from palinode.cli import main


def test_main_help_names_palinode_dir_and_its_fallback() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    normalized_help = " ".join(result.output.split())
    assert "PALINODE_DIR names the memory directory and the palinode.config.yaml read from it" in normalized_help
    assert "Unset, it is ~/palinode" in normalized_help
