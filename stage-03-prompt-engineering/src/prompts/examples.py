"""
Ready-to-use prompt templates.

INVOICE_EXTRACTOR — your contribution
──────────────────────────────────────
The system prompt is a stub. Writing it is Exercise 1 of this stage.
See the docstring guidance and the test in test_template.py for what
the model needs to extract: <vendor>, <date>, <amount>, <invoice_number>.
"""

from prompts.template import PromptTemplate

# ── Invoice extraction ────────────────────────────────────────────────────────

INVOICE_EXTRACTOR = PromptTemplate(
    name="invoice-extractor",
    version="2026-01-01",
    description="Extract structured fields from invoice text using XML tags.",
    system="""
TODO: Write a system prompt that reliably extracts invoice fields.

The model must respond with exactly these four XML tags:
  <vendor>      the business name on the invoice
  <date>        invoice date in ISO format (YYYY-MM-DD)
  <amount>      total amount due, numeric only, no currency symbol
  <invoice_number> the invoice reference/PO number

Consider including:
  - A clear role ("You are an invoice parsing assistant…")
  - Explicit output format instructions showing the exact tags
  - A rule for handling missing fields (e.g. output <date>UNKNOWN</date>)
  - At least one worked example (few-shot inside the system prompt)

Tip: structured output via XML is more reliable than asking for JSON in free
text — use the tags exactly as listed above.
""".strip(),
    user_template="Extract the invoice details from this document:\n\n{document}",
)


# ── Sentiment classifier ──────────────────────────────────────────────────────

SENTIMENT_CLASSIFIER = PromptTemplate(
    name="sentiment-classifier",
    version="2026-01-01",
    description="Classify text sentiment as positive, negative, or neutral.",
    system="""\
You are a sentiment analysis assistant. Classify the sentiment of the given text.

Respond with exactly one XML tag:
  <sentiment>positive</sentiment>
  <sentiment>negative</sentiment>
  <sentiment>neutral</sentiment>

Rules:
  - Use only the three values above, lowercase.
  - Base your judgment on the overall tone, not individual words.
  - Mixed signals → neutral.""",
    user_template="Classify the sentiment of this text:\n\n{text}",
)


# ── Structured summariser ─────────────────────────────────────────────────────

STRUCTURED_SUMMARISER = PromptTemplate(
    name="structured-summariser",
    version="2026-01-01",
    description="Summarise a document into title, key points, and one-line tldr.",
    system="""\
You are a document summarisation assistant. Summarise the provided document.

Respond using exactly these XML tags:
  <title>   a short descriptive title (max 10 words)
  <tldr>    a single sentence summary
  <points>  bullet points — one <point> tag per key idea (3–5 points)

Example format:
<title>Q3 Sales Performance Review</title>
<tldr>Revenue grew 12% YoY driven by enterprise deals despite slowing SMB.</tldr>
<points>
<point>Enterprise segment up 28% — three new logo deals above $500k.</point>
<point>SMB churn increased to 4.2%, up from 2.8% in Q2.</point>
<point>APAC expansion contributed $1.2M for the first time.</point>
</points>""",
    user_template="Summarise this document:\n\n{document}",
)
