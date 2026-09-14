"""A grounded question-answering feature for the dashboard: lets a viewer
ask a plain-language question about whatever real numbers are currently
on screen, and get an answer that can only draw on those real, already-
computed numbers, never on general knowledge or an invented figure.

This is deliberately not a general chatbot bolted onto the app. The
system prompt instructs the model to answer only from the real context
block built in build_context() below, and to say so plainly if a question
asks about something that context does not cover, the same "disclose the
real gap rather than invent an answer" discipline used everywhere else in
this project, just applied to a conversational interface instead of a
chart caption.

Bring-your-own-key, the same real pattern already live on this project's
sibling (the battery-electrode-screening-agent's own demo): a shared
server key would mean one visitor's questions bill another visitor's
account, so each visitor supplies their own Anthropic API key, used for
one call and never stored.
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are answering questions about a live materials flow analysis dashboard. "
    "You will be given a block of real, already-computed numbers from the dashboard's current "
    "state. Answer strictly from those numbers. Never invent a figure that is not given, and "
    "never fall back on general knowledge about materials or markets to fill a gap. If the "
    "question asks about something the provided numbers do not cover, say so plainly rather "
    "than guessing. Keep answers to two to four sentences, plain language, no jargon that was "
    "not already used in the numbers you were given."
)


def build_context(**kwargs: object) -> str:
    """Assemble the real, currently-on-screen numbers into a labeled text block, one section
    per real quantity, using None-checks so a scenario missing some numbers (e.g. no connector
    result for a no-exports disruption scenario) still produces an honest, partial context
    instead of crashing on a missing key."""
    lines: list[str] = []

    def add(label: str, value: object) -> None:
        if value is not None:
            lines.append(f"{label}: {value}")

    add("Selected material", kwargs.get("material_label"))
    add("Real physical system", kwargs.get("system_label"))
    add("Projection horizon (year)", kwargs.get("horizon_year"))
    lines.append("")
    lines.append("Real 2023 UN Comtrade trade concentration:")
    add("  HHI (pooled)", kwargs.get("hhi"))
    add("  Real largest supplier", kwargs.get("top1_country"))
    add("  That supplier's real share of trade", kwargs.get("top1_share"))
    add("  Disruption scenario: supplier removed", kwargs.get("removed_country"))
    add("  That supplier's real share of trade", kwargs.get("removed_country_share"))
    # The real slack percentage varies per material now (copper: a real, cited 22.4%; every
    # other material: the same disclosed 20% assumption crm-trade-network's own connector
    # already uses), so the label itself is built from the real value passed in rather than a
    # hardcoded "20%" that would be wrong for copper specifically.
    slack_label = kwargs.get("real_slack_pct") or "real spare capacity"
    add("  Real shortfall, no substitution (1 year)", kwargs.get("no_slack_shortfall"))
    add(f"  Real shortfall, {slack_label} slack (1 year)", kwargs.get("slack_shortfall"))
    lines.append("")
    lines.append("Physical recovery from the real material flow projection:")
    add("  Average annual recovery (USD)", kwargs.get("avg_annual_usd"))
    add("  Peak retirement year", kwargs.get("peak_year"))
    add("  Recovery in that peak year (USD)", kwargs.get("peak_usd"))
    add("  Coverage of no-substitution shortfall", kwargs.get("coverage_no_slack"))
    add(f"  Coverage of {slack_label}-slack shortfall", kwargs.get("coverage_20pct"))
    add("  Price used (USD/tonne)", kwargs.get("price"))
    add("  Price source", kwargs.get("price_source"))
    if kwargs.get("price_range"):
        lines.append(f"  Real price range and its coverage sensitivity: {kwargs['price_range']}")
    lines.append("")
    lines.append("Minimum-cost supplier diversification (a linear program, see the dashboard's own caption):")
    add("  Target: max share for any one supplier", kwargs.get("target_share"))
    add("  Real reallocation needed, no substitution", kwargs.get("div_no_slack_note"))
    add(f"  Real reallocation needed, {slack_label} slack", kwargs.get("div_slack_note"))
    if kwargs.get("cross_material_table"):
        lines.append("")
        lines.append("How this material compares to the other six (each against its own real top supplier):")
        lines.append(str(kwargs["cross_material_table"]))

    return "\n".join(lines)


def ask_dashboard(api_key: str, question: str, context: str) -> str:
    """Send one grounded question to Claude. Raises on any real API error (bad key, network,
    rate limit) rather than swallowing it, so the caller can show the real error to the viewer
    instead of a silent wrong answer."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Real dashboard numbers:\n\n{context}\n\nQuestion: {question}"}],
    )
    return response.content[0].text
