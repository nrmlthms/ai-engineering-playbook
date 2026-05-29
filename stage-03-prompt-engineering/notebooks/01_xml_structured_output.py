# ruff: noqa: F704, E402
# %% [markdown]
# # 01 — XML Structured Output
#
# Reliable structured output from LLMs without JSON fragility.
# No live API calls in sections 1–3 (pure parsing demos).
# Section 4 requires ANTHROPIC_API_KEY.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path().resolve().parent / "src"))

from extractor import assert_tags_present, extract_all_tags, extract_tag, extract_tags

# %% [markdown]
# ## 1. Why XML tags beat raw JSON
#
# The model may add prose, markdown fences, or comments around JSON.
# XML tags are punctuation the model treats as delimiters — they almost
# never appear inside natural prose, so extraction is unambiguous.

# %%
# Simulated messy model output — common in the wild
messy_json = """
Sure! Here is the data you asked for:

```json
{
  "name": "Acme Corp",
  "amount": 1250.00,  // including tax
  "date": "Jan 15 2025"
}
```

Let me know if you need anything else!
"""

# Compare: clean XML output from a well-prompted model
clean_xml = """
<vendor>Acme Corp</vendor>
<amount>1250.00</amount>
<date>2025-01-15</date>
<invoice_number>INV-00423</invoice_number>
"""

result = extract_tags(clean_xml, ["vendor", "date", "amount", "invoice_number"])
print("Extracted fields:")
for k, v in result.items():
    print(f"  {k}: {v!r}")

# %% [markdown]
# ## 2. Tag extraction patterns

# %%
# Single tag — returns first match or None
text = "The capital of France is <answer>Paris</answer>."
print(extract_tag(text, "answer"))          # "Paris"
print(extract_tag(text, "missing"))         # None

# All tags — list of all matches
steps = "<step>Load data</step><step>Clean data</step><step>Train model</step>"
print(extract_all_tags(steps, "step"))

# Multiple tags in one pass
invoice = """
<vendor>TechSupplies Ltd</vendor>
<date>2025-03-22</date>
<amount>4850.00</amount>
<invoice_number>TS-2025-0042</invoice_number>
"""
fields = extract_tags(invoice, ["vendor", "date", "amount", "invoice_number"])
print(fields)

# %% [markdown]
# ## 3. Handling missing fields
#
# `assert_tags_present()` turns a soft None into a hard error at the boundary
# where your application needs all fields. Use it after extraction, not inside.

# %%
# Partial extraction — model missed the date
partial = "<vendor>Acme Corp</vendor><amount>500.00</amount>"
result = extract_tags(partial, ["vendor", "date", "amount"])

try:
    assert_tags_present(result, ["vendor", "date", "amount"])
except ValueError as e:
    print(f"Validation error: {e}")

# Successful validation
complete = "<vendor>Acme Corp</vendor><date>2025-01-15</date><amount>500.00</amount>"
result = extract_tags(complete, ["vendor", "date", "amount"])
assert_tags_present(result, ["vendor", "date", "amount"])
print("All required fields present:", result)

# %% [markdown]
# ## 4. Live: invoice extraction (requires ANTHROPIC_API_KEY)
#
# This uses `INVOICE_EXTRACTOR` from `prompts/examples.py`.
# **The system prompt is a stub — complete Exercise 1 first.**

# %%
SAMPLE_INVOICE = """
INVOICE

From: TechSupplies Ltd
      42 Innovation Drive, London, EC1A 1BB

To:   Acme Corporation
      1 Main Street, New York, NY 10001

Invoice Number: TS-2025-0042
Invoice Date:   22 March 2025
Due Date:       22 April 2025

Description                  Qty   Unit Price    Total
────────────────────────────────────────────────────────
Developer laptop (ThinkPad)   2     £1,800.00   £3,600.00
USB-C dock                    5       £120.00     £600.00
Extended warranty (2yr)       2       £325.00     £650.00
────────────────────────────────────────────────────────
                                   TOTAL DUE:  £4,850.00

Payment terms: 30 days. Bank transfer preferred.
"""


async def extract_invoice(invoice_text: str) -> dict[str, str | None]:
    """Extract structured fields from invoice text using AnthropicClient."""
    from llm.anthropic_client import AnthropicClient
    from llm.sampling import SamplingParams
    from prompts.examples import INVOICE_EXTRACTOR

    client = AnthropicClient()
    system, user = INVOICE_EXTRACTOR.render(document=invoice_text)

    response = await client.complete(
        messages=[{"role": "user", "content": user}],
        system=system,
        params=SamplingParams(temperature=0.0, max_tokens=512),
    )

    fields = extract_tags(response.content, ["vendor", "date", "amount", "invoice_number"])
    print(f"Model: {response.model}  |  Cost: ${response.cost.total_usd:.4f}" if response.cost else "")
    print("Raw response:\n", response.content)
    return fields


# Uncomment after completing Exercise 1 (writing the system prompt):
# result = await extract_invoice(SAMPLE_INVOICE)
# print("\nExtracted fields:")
# for k, v in result.items():
#     print(f"  {k}: {v!r}")
print("(Cell ready — complete Exercise 1 then uncomment the last lines)")
