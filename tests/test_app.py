from __future__ import annotations

import pytest

pytest.importorskip("gradio")

from app import render_suspect_profiles
from case_file import CaseFile, Claim, ClaimStatus, SuspectProfile


def test_render_suspect_profiles_displays_supported_and_unknown_claims() -> None:
    case_file = CaseFile(mystery_text="A necklace vanished from the study overnight.")
    case_file.suspect_profiles = [
        SuspectProfile(
            suspect="The Housekeeper",
            motive=(
                Claim(statement="Owed the victim money.", status=ClaimStatus.SUPPORTED, evidence_ids=("E-02",)),
            ),
            opportunity=(
                Claim(statement="No record places her near the study.", status=ClaimStatus.UNKNOWN),
            ),
        )
    ]

    markdown = render_suspect_profiles(case_file)

    assert "The Housekeeper" in markdown
    assert "Owed the victim money. (E-02)" in markdown
    assert "_unknown:_ No record places her near the study." in markdown


def test_render_suspect_profiles_before_analysis_shows_placeholder() -> None:
    case_file = CaseFile(mystery_text="A necklace vanished from the study overnight.")

    assert render_suspect_profiles(case_file) == "_No suspect profiles yet._"
