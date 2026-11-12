# Clinic SOP RAG

Ask plain-English questions about clinic standard operating procedures, practice policies and compliance guidance, and get answers grounded in the source PDFs, with citations to the file and page.

## The problem

Outpatient clinics run on a large body of written procedure: appointment and scheduling policies, financial and billing policies, referral and test-tracking procedures, and federal compliance guidance such as the HHS OIG compliance program guidance. These documents are:

- **Scattered** across many PDFs from different organisations (AAP, OIG, NOSORH, ACR and others).
- **Long and dense.** The OIG General Compliance Program Guidance alone runs to dozens of pages.
- **Hard to search.** Keyword search misses answers phrased differently from the question ("after-hours appointments" vs. "extended hours of care").

Front-desk staff, practice managers and compliance officers need a quick, reliable answer to questions like *"What are our extended hours?"* or *"What are the seven elements of an effective compliance program?"*, and they need to know **which document the answer came from** so they can check it.

A general-purpose chatbot isn't enough: it doesn't know a specific practice's policies and may produce confident but wrong answers.

## The solution

This project is a **Retrieval-Augmented Generation (RAG)** pipeline:

1. **Ingest:** load every PDF under `data/` and split it into pages.
2. **Embed:** turn each page into a 384-dimensional vector with the `all-MiniLM-L6-v2` sentence-transformer model, which runs locally.
3. **Store:** save the vectors, the text and the metadata (source file, page, title) in a persistent **ChromaDB** vector database.
4. **Retrieve:** embed the user's question the same way and fetch the most similar pages.
5. **Generate:** send the question and the retrieved pages to an LLM hosted on **Groq**. The LLM is told to answer **only from that context**, cite `[file, page N]`, and say so when the documents don't contain the answer.

```
 PDFs ──► PyMuPDF loader ──► MiniLM embeddings ──► ChromaDB
                                                       │
 Question ──► MiniLM embedding ──► similarity search ──┘
                                        │
                              top-k pages + metadata
                                        │
                                        ▼
                         Groq LLM (grounded prompt) ──► Answer + sources
```

### Example

```
> python generate_embeddings.py "What is the policy for extended hours of care?"

Answer:
The practice offers extended after-work and weekend hours:
- Monday – Friday: 8 am – 9 pm
- Saturday: 8 am – 2 pm
For care outside these hours, patients should refer to the "After Hours Phone
Calls and Web Messaging" policy ...  [Extended-Hours-of-Care.pdf, page 1]

Sources:
  - data\aap_policy_templates\Extended-Hours-of-Care.pdf (page 1), score 0.766
  - data\aap_policy_templates\PT_MP_Appointment_Policy.pdf (page 1), score 0.540
  ...
```

## Document corpus

31 publicly available PDFs (256 pages), listed with their origins in [data/SOURCES.md](data/SOURCES.md):

| Folder | Content |
|---|---|
| `aap_policy_templates/` | American Academy of Pediatrics practice policy templates (appointments, billing, referrals, test tracking, team huddles, …) |
| `oig_compliance/` | HHS OIG compliance program guidance: general (2023), physician practices, third-party billing, clinical labs |
| `nosorh_rhc/` | NOSORH Rural Health Clinic scheduling and staffing guides |
| `acr_ultrasound/` | American College of Radiology ultrasound quality-control and accreditation requirements |
| `aapc_coding_alerts/` | AAPC coding alert on OIG guidance for physician practices |
| `ttuhsc_el_paso/` | TTUHSC El Paso patient access centre policy |

## Project structure

| File | Purpose |
|---|---|
| `documents_structure.py` | Loads every PDF under `data/` with PyMuPDF (one LangChain `Document` per page, with metadata). |
| `chunk_conversion.py` | Splits documents into overlapping chunks (1000 characters, 200 overlap) with `RecursiveCharacterTextSplitter`. |
| `generate_embeddings.py` | The main pipeline: `embeddingManager` (MiniLM), `VectorStore` (ChromaDB), `RAGRetriever` (similarity search), `GroqLLM` (answer generation), plus the command-line entry point. |
| `inspect_db.py` | Exports the vector database (ids, text, metadata, embeddings) to `data/collection_export.csv` for inspection. |
| `data/` | Source PDFs and `SOURCES.md`. The vector database is written to `data/db/`, which git ignores. |

## Setup

Requires **Python 3.12**.

```powershell
py -V:3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the project root with a Groq API key (free at https://console.groq.com/keys):

```
GROQ_API_KEY=gsk_your_key_here
# optional, defaults to openai/gpt-oss-20b
GROQ_MODEL=openai/gpt-oss-20b
```

## Usage

```powershell
# Ask a single question
python generate_embeddings.py "What are the seven elements of an effective compliance program?"

# Interactive mode: ask several questions, blank line to quit
python generate_embeddings.py
```

On the **first run**, the script builds the vector database in `data/db/`. Later runs reuse it and go straight to answering. To rebuild, for example after adding PDFs, delete `data/db/` and run again.

Optional: set `$env:HF_HUB_OFFLINE = "1"` after the embedding model has been downloaded once, to skip Hugging Face network checks on startup.

### Inspecting the database

```powershell
python inspect_db.py              # export all records to data/collection_export.csv
python inspect_db.py --show 3     # also print 3 records in the terminal
```

You can also open `data/db/chroma.sqlite3` with a SQLite viewer. Metadata lives in the `embedding_metadata` table, one row per field, and the page text is stored under the key `chroma:document`.

## Design notes and issues solved

- **Answers grounded in the documents.** The prompt restricts the LLM to the retrieved context and requires citations. When the context doesn't cover a question, the model says so instead of guessing, which matters for compliance-related answers.
- **No repeated embedding.** Ingestion only runs when the collection is empty, so re-running the script never re-embeds or duplicates documents. An earlier version had all its code at module level, so importing it, or running it as `python -m generate_embeddings.py`, re-ran ingestion and doubled the database. The run code now sits behind `if __name__ == "__main__":`.
- **Correct similarity scores.** ChromaDB's default distance is squared L2, not cosine. Because MiniLM produces unit-length vectors, `L2² = 2 − 2·cos`, so the retriever converts with `similarity = 1 − distance / 2`. This gives true cosine similarity, which makes `score_threshold` meaningful. If the collection is rebuilt with `"hnsw:space": "cosine"`, change this back to `1 − distance`.
- **Current LangChain and Groq APIs.** Prompt and message classes are imported from `langchain_core`, since `langchain.prompts` and `langchain.schema` were removed in LangChain 1.x. The default model is `openai/gpt-oss-20b`, because Groq retired older models such as `gemma2-9b-it` and `llama-3.3-70b-versatile`. Override it with `GROQ_MODEL`.
- **Secrets stay local.** The API key is read from `.env`, which is git-ignored, and is never printed.

## Limitations and next steps

- **Page-level retrieval.** The database currently stores whole pages (256 entries). The chunks from `chunk_conversion.py` (993 of them) are computed but not yet stored. Switching to chunks should improve precision, especially for long documents where the answer is spread across a page that covers many topics.
- **Cosine collection.** When rebuilding, create the collection with `"hnsw:space": "cosine"`.
- **No conversation memory.** Each question is answered on its own.
- **Interface.** Command line only for now. A web interface (for example Streamlit) and structured report filling (such as incident reports or audit checklists) are planned extensions.

## Disclaimer

This is a learning and demonstration project built on public sample policies and government guidance. It is not legal, compliance or medical advice. **Do not send patient data (PHI) to it.** Questions and retrieved text go to Groq's hosted API, and HIPAA-regulated use would need a provider covered by a Business Associate Agreement, or a locally hosted model.
