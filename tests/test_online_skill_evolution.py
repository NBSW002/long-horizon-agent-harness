import asyncio

from agents.online_skill_evolution import extract_online_skill_candidate


def test_extractor_rejects_candidate_without_reusable_evidence():
    async def side_query(_system: str, _prompt: str) -> str:
        return (
            '{"skills": [{"name": "formatting", "description": "Format reports", '
            '"instructions": "Use a clear report structure."}]}'
        )

    candidate = asyncio.run(
        extract_online_skill_candidate(
            messages=[{"role": "user", "content": "Please always use this format."}],
            side_query=side_query,
        )
    )

    assert candidate is None


def test_extractor_keeps_candidate_with_explicit_evidence():
    async def side_query(_system: str, _prompt: str) -> str:
        return (
            '{"skills": [{"name": "formatting", "description": "Format reports", '
            '"instructions": "Use a clear report structure.", '
            '"evidence": "The user explicitly corrected the previous report structure."}]}'
        )

    candidate = asyncio.run(
        extract_online_skill_candidate(
            messages=[{"role": "user", "content": "Please always use this format."}],
            side_query=side_query,
        )
    )

    assert candidate is not None
    assert candidate.evidence.startswith("The user explicitly")
