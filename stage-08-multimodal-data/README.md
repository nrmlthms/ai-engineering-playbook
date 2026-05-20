# Stage 08 — Multimodal & Data

## Concepts

- Vision: image inputs, base64 encoding, Claude's vision capabilities
- Document understanding: PDFs, tables, charts
- Structured data extraction: receipts, forms, invoices
- Batch processing with the Anthropic Batch API
- Cost and latency trade-offs for multimodal inputs

## Key ideas

```
Image → base64 encode → messages[].content[{type: image}]
                                          ↓
                              Model sees pixels as tokens
                              (~1 token per 750 px²)
```

Images are expensive — a 1080p screenshot can cost ~1500 tokens. Resize and
crop before sending; extract text with OCR first if the task is text-only.

## What's in `src/`

| File | Purpose |
|------|---------|
| `vision.py` | Send images to Claude, parse structured output |
| `pdf.py` | Extract text + tables from PDFs |
| `batch.py` | Batch API wrapper for high-throughput extraction |

## Exercises

1. Extract line items from a photo of a receipt into a Pydantic model
2. Process 100 invoices via the Batch API and measure cost vs streaming
3. Build a chart-to-data extractor that outputs CSV
