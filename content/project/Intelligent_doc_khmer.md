---
title: "Intelligent Document Processing for Khmer | Applied AI Research Internship"
date: "2026-08-31"
excerpt: "An applied AI research internship at Techo Startup Center focused on Khmer document understanding: multimodal RAG, OCR evaluation, multilingual embeddings, and table-structure detection."
tags: ["Python", "PyTorch", "Khmer NLP", "Low-Resource Languages", "OCR", "RAG", "Vision-Language Models", "Qwen3-VL", "YOLO26", "Document AI"]
thumbnail: "/images/projects/khmer_idp/background.jpg"
---

# Intelligent Document Processing for Khmer Language

**Applied AI research internship · Techo Startup Center · Phnom Penh, Cambodia · June–August 2026**

> **The central question:** How can document AI systems extract and retrieve reliable information from Khmer documents when OCR support, annotated datasets, and language-specific evaluation resources remain limited?

| 892 | 1,000+ | 90% | 98.53% |
|:--|:--|:--|:--|
| Khmer tables in the synthetic benchmark | Table images manually annotated | Relevant page retrieved at Hit@1 | YOLO26s mAP@0.5 for row/column detection |


<div align="center">
  <table>
    <tbody>
      <tr>
        <td align="center" width="50%">
          <img
            src="/images/projects/khmer_idp/khmer_post.jpg"
            alt="Example of a Khmer document"
            width="100%"
          />
        </td>
        <td align="center" width="50%">
          <img
            src="/images/projects/khmer_idp/khmer_table.png"
            alt="Example of a Khmer table"
            width="100%"
          />
        </td>
      </tr>
      <tr>
        <td align="center">
          <em>Example of a Khmer document</em>
        </td>
        <td align="center">
          <em>Example of a Khmer table</em>
        </td>
      </tr>
    </tbody>
  </table>
</div>


**At a glance.** During my AI Research Internship at Techo Startup Center, I explored and evaluated document-processing models for Khmer, developed a multimodal Retrieval-Augmented Generation (RAG) pipeline for question answering over Khmer PDFs, compared multilingual embedding models, and prepared datasets for table understanding. The work combined practical system development with benchmark construction and quantitative evaluation.

> **Reading guide:** This case study is organized around the problem, system design, experiments, results, and lessons learned. Metrics are reported from my internship report; image paths are placeholders to replace with figures exported and approved for the portfolio.

---

## 1. Why Khmer document understanding?

![Khmer character](/images/projects/khmer_idp/khmer_character.png)

Modern document AI systems often perform better on widely represented languages such as English than on languages with fewer digital resources. Khmer documents also contain complex visual structures: text, tables, charts, figures, and multi-column layouts. Recognizing individual words is not enough if the relationships between rows, columns, labels, and values are lost.

These issues affect downstream question answering. OCR errors, missing table cells, or incorrect reading order can prevent the retrieval system from finding the right evidence and can reduce the reliability of generated answers.

The internship therefore addressed three connected challenges:

- **Document parsing:** extract Khmer text and document structure from PDFs.
- **Grounded question answering:** retrieve relevant content and combine OCR text with visual evidence.
- **Table understanding:** evaluate table extraction and train a specialized detector for rows and columns.

---

## 2. The overall workflow

The work moved from model exploration to a question-answering prototype and dedicated table experiments.

```text
Khmer PDFs and table images
          |
          v
Explore document parsers
(PP-StructureV3, PaddleOCR-VL, Surya OCR 2)
          |
          +------------------------------+
          |                              |
          v                              v
Multimodal RAG pipeline           Table datasets
OCR + chunking                    Manual row/column annotation
BGE-M3 embeddings                 892-table Khmer benchmark
FAISS retrieval                   Surya OCR 2 evaluation
Qwen3-VL answer generation        YOLO26s fine-tuning
          |                              |
          v                              v
Retrieval and answer metrics      Structure detection metrics
```

The final RAG pipeline used **Surya OCR 2**, recursive text chunking, **BAAI/bge-m3** embeddings, **FAISS** dense retrieval, and **Qwen3-VL 4B** for answer generation.

---

## 3. Project A — Multimodal question answering over Khmer PDFs

### 3.1 System design

I developed a multimodal Retrieval-Augmented Generation (RAG) pipeline to answer questions about Khmer PDF documents. The evaluation dataset covered 10 pages from two sources: a 2015 KhmerPost newspaper and the National Bank of Cambodia Annual Report 2024.

I prepared 40 question–answer pairs generated by an LLM and manually verified them. The system was evaluated at two levels: retrieving the relevant source pages and generating answers from those pages.

The pipeline consisted of:

1. **OCR and layout extraction:** Extract text and document structure using Surya OCR 2.
2. **Khmer-aware chunking:** Split extracted text into manageable chunks while preserving meaningful text boundaries.
3. **Embedding:** Encode document chunks using BAAI/bge-m3.
4. **Dense retrieval:** Index normalized embeddings in FAISS and retrieve relevant chunks using inner-product similarity.
5. **Page-level source selection:** Group retrieved chunks by their original pages and select the highest-ranked page.
6. **Answer generation:** Use Qwen3-VL 4B in vision-only or hybrid mode.
7. **Evaluation:** Measure retrieval performance and assess generated answers using automatic metrics and an LLM judge.

The system was deployed through a Gradio interface in Google Colab.

![Multimodal RAG pipeline](/images/projects/khmer_idp/rag_pipeline.png)

### 3.2 Khmer-aware retrieval implementation

Khmer text requires careful chunking because splitting text at arbitrary character positions can separate meaningful linguistic units. I used recursive character-based chunking with the following configuration:

| Parameter | Configuration |
|:--|--:|
| Chunk size | 1,200 characters |
| Chunk overlap | 50 characters |
| Embedding model | BAAI/bge-m3 |
| Document embedding limit | 1,024 tokens |
| Question embedding limit | 512 tokens |
| Vector index | FAISS IndexFlatIP |

The splitting priority was:

1. Paragraph boundaries
2. Line breaks
3. Khmer sentence punctuation
4. Fixed-length character splitting as a fallback

The resulting chunks were embedded and indexed using normalized BGE-M3 embeddings. FAISS `IndexFlatIP` performed dense similarity search, after which retrieved chunks were grouped back into page-level candidates.

This design connected text-based retrieval with visual document understanding: the retrieval component identified a relevant page, and Qwen3-VL used that page to answer the user's question.

### 3.3 Retrieval performance

I evaluated retrieval using 40 manually verified questions. Two complementary metrics were used:

- **Page Hit@k:** Whether the correct source page appeared among the top \(k\) retrieved results.
- **Document Hit@k:** Whether the correct source document appeared among the top \(k\) results.
- **Page MRR@k:** The mean reciprocal rank of the correct page, rewarding results that place the relevant page closer to the top.

| Top-k | Page Hit@k | Document Hit@k | Page MRR@k |
|--:|--:|--:|--:|
| 1 | 90.0% | 100% | 0.9000 |
| 2 | 92.5% | 100% | 0.9125 |
| 3 | 95.0% | 100% | 0.9208 |
| 4 | 97.5% | 100% | 0.9271 |
| 5 | 100% | 100% | 0.9321 |

The correct page ranked first for **90% of the questions**, and appeared within the top five for all 40 questions. The correct source document was identified for every question, even at top-1.

The results suggest that dense retrieval worked well on these selected documents. However, identifying the correct document was easier than identifying the exact page within it, highlighting the importance of page-level retrieval for document question answering.

These findings are specific to the evaluated dataset and do not establish general retrieval performance across Khmer documents.

### 3.4 Does combining text and images improve answers?

I compared two answer-generation strategies using the same top-ranked page:

- **Vision-only:** Qwen3-VL received the page image.
- **Hybrid:** Qwen3-VL received the page image together with text extracted by Surya OCR 2.

Seven questions were excluded from answer-quality evaluation because of CUDA out-of-memory errors, leaving 33 questions.

The evaluation considered answer similarity, context relevance, answer correctness, and total response time.

| Metric | Hybrid: OCR text + image | Vision-only |
|:--|--:|--:|
| Page Hit@1 | 87.88% | 87.88% |
| Answer similarity | 0.7063 | 0.6656 |
| Context relevance | 0.5739 | 0.5739 |
| Answer correctness | 0.7580 | 0.7295 |
| Average response time | 17.86 s | 13.42 s |

The hybrid approach achieved higher answer similarity and correctness, while context relevance remained unchanged. Both approaches had the same reported Page Hit@1 on this subset.

Compared with vision-only generation, the hybrid approach improved:

- **Answer similarity:** approximately 6.1% relative improvement.
- **Answer correctness:** approximately 3.9% relative improvement.

These relative improvements are calculated from the reported metric values; they should not be interpreted as percentage-point increases in accuracy.

The trade-off was response time: hybrid generation took approximately 4.44 seconds longer on average, or about 33% more time.

Overall, the results suggest that OCR-derived text can complement visual evidence when answering questions about Khmer documents, although the limited evaluation size calls for further validation.

### 3.5 LLM-as-judge evaluation

In addition to the automatic metrics, I used Qwen3-VL 4B as an LLM judge to assess generated answers on a 0–5 scale across four dimensions:

- **Correctness:** Whether the answer accurately addresses the question.
- **Completeness:** Whether the answer includes the necessary information.
- **Groundedness:** Whether the answer is supported by the available evidence.
- **Khmer quality:** The quality of the generated Khmer response.

| Evaluation dimension | Hybrid | Vision-only |
|:--|--:|--:|
| Correctness | 4.09/5 | 3.27/5 |
| Completeness | 4.09/5 | 3.18/5 |
| Groundedness | 4.09/5 | 3.67/5 |
| Khmer quality | 4.09/5 | 3.82/5 |

The LLM-as-judge results also favored the hybrid approach across all four dimensions. The largest difference was in correctness, followed by completeness.

This additional evaluation provides a more nuanced view than answer similarity alone: two answers can use different wording while still being correct, complete, and grounded in the source. However, because the judge was itself a language model, these scores should be treated as complementary assessments rather than independent proof of answer quality.

### 3.6 Key findings and limitations

The experiments highlighted three main findings:

1. **Effective page retrieval:** The correct page was ranked first for 90% of the 40 questions and appeared in the top five for all questions.
2. **Benefits of multimodal evidence:** Combining OCR text with the page image produced better answer-quality scores than using the image alone in the evaluated subset.
3. **A quality–latency trade-off:** Hybrid generation improved the reported quality metrics but required more processing time.

There are also important limitations. The retrieval evaluation covered only 10 pages from two documents, and the answer-quality comparison used 33 questions after seven were excluded due to GPU memory constraints. The automatic and LLM-judge evaluations also rely on metrics that have their own limitations.

Future work should expand the dataset to more documents, page layouts, and question types; reduce GPU memory requirements; and validate answer quality using a larger set of human assessments.

This project gave me practical experience building and evaluating a multimodal RAG system for a low-resource language, from Khmer-aware text processing and dense retrieval to vision-language answer generation and systematic evaluation.

---

## 4. Project B — Evaluating table understanding

Tables are especially challenging for document AI. A system may recognize the text in some cells correctly but still fail to recover the complete table if rows, columns, or cells are omitted.

I worked on two complementary datasets:

- **More than 1,000 table images** were manually annotated with row and column bounding boxes using **LabelMe**. Approximately 250 contained Khmer tables.
- A separate benchmark of **892 rendered Khmer tables** was created with structured HTML ground truth. The workflow used Nemotron-Parse to structure English source tables, translated their text into Khmer while preserving the table structure, normalized the HTML, and rendered the tables as images.

![Table annotation example](/images/projects/khmer_idp/table_annotation.png)

### 4.1 Surya OCR 2: text accuracy versus structural completeness

On the 892-table benchmark, Surya OCR 2 produced the following global results:

| Metric | Result |
|:--|--:|
| Ground-truth cells | 323,018 |
| Predicted cells | 141,818 |
| Matched cells | 81,735 |
| Exact accuracy on matched cells | 96.58% |
| Character Error Rate on matched cells | 2.74% |
| Global missing-cell rate | 74.70% |

The key finding was the difference between **recognition accuracy for matched cells** and **complete table recovery**. Text was generally accurate when a cell was successfully matched, but many ground-truth cells were missing from the predictions.

The missing-cell rate also increased with table size:

| Table size | Cell coverage | Cell recall | Missing-cell rate |
|:--|--:|--:|--:|
| Small | 84.50% | 53.97% | 46.03% |
| Medium | 48.52% | 27.83% | 72.17% |
| Large | 33.80% | 22.10% | 77.90% |

This highlighted a major limitation: long or complex tables were often only partially extracted. Strong text accuracy on the cells that were recovered did not guarantee that the full table was represented.

### 4.2 Row and column detection with Surya OCR 2

I also evaluated Surya OCR 2's row and column bounding boxes against a manually annotated **English table subset**. This was a separate evaluation from the 892 rendered Khmer-table benchmark.

| Metric | Result |
|:--|--:|
| Average row IoU | 0.6435 |
| Average column IoU | 0.6197 |
| Average overall IoU | 0.6316 |
| Row recall at IoU 0.5 | 79.01% |
| Column recall at IoU 0.5 | 72.52% |

The results showed that Surya OCR 2 recovered a substantial portion of the table structure, although row and column localization remained imperfect.

### 4.3 Fine-tuning YOLO26 for table structure detection

To investigate a more specialized approach, I fine-tuned YOLO26s and YOLO26m to detect table rows and columns using the manually annotated dataset.

| Metric | YOLO26s | YOLO26m |
|:--|--:|--:|
| mAP@0.5 | 98.53% | 98.57% |
| mAP@0.5:0.95 | 86.66% | 86.65% |
| Precision | 96.66% | 96.71% |
| Recall | 96.37% | 96.43% |
| Inference time per image | 11.16 ms | 26.81 ms |
| Model size | 19.38 MB | 41.99 MB |

YOLO26s achieved nearly the same detection metrics as YOLO26m while using a smaller model and running faster in these experiments. It therefore offered the more favorable balance of accuracy, model size, and inference speed for this task.

The results suggest a **modular approach**: use a general document parser for OCR and document extraction, then add a task-specific detector when more precise row and column localization is needed.

---

## 5. Tools and technologies

**Programming and deep learning:** Python, PyTorch, NumPy, Pandas  
**Document AI and vision:** Surya OCR 2, PP-StructureV3, PaddleOCR-VL, Qwen3-VL, YOLO26, OpenCV, Pillow  
**Retrieval and embeddings:** BAAI/bge-m3, FAISS, recursive text chunking  
**Data and prototyping:** LabelMe, Gradio, Hugging Face, Ultralytics, Playwright, Chromium  
**Evaluation:** Hit@k, MRR, answer similarity, answer correctness, CER, IoU, precision, recall, mAP

---

## 6. Challenges and lessons learned

### Limited language-specific resources

Publicly available annotated Khmer datasets for document layout, table extraction, chart understanding, and question answering were limited. Dataset preparation and custom evaluation were therefore important parts of the work, rather than merely preliminary steps.

### Document structure matters as much as recognized text

A table can contain correctly recognized words and numbers but still lose its meaning if the relationships between cells are not preserved. This was visible in the gap between Surya OCR 2's matched-cell accuracy and its global missing-cell rate.

### Model choice depends on the task

A general document parser and a specialized object detector solve different problems. Surya OCR 2 provided OCR and document parsing, while YOLO26s was more effective for the narrowly defined row/column detection task in the reported experiments.

### Quality and efficiency must be considered together

The hybrid RAG approach improved the measured answer-quality metrics, but increased average response time. Likewise, YOLO26m offered marginally different detection metrics from YOLO26s but required more inference time and a larger model.

---

## 7. Future directions

The report identified several directions for further work:

1. Expand the diversity and scale of annotated Khmer document datasets.
2. Combine YOLO26s structural detection with OCR-based cell-content extraction to improve table reconstruction.
3. Preserve page, section, region, table, and figure metadata for more layout-aware retrieval and source traceability.
4. Improve chart understanding beyond extracting visible labels.
5. Build a larger, manually validated Khmer-specific benchmark for semantic embeddings.

---

## 8. Key takeaways

| Finding | Evidence | Lesson |
|:--|:--|:--|
| Retrieval was strong on the evaluated questions | 90% Hit@1; 100% Hit@5 | Semantic retrieval can locate relevant pages in this selected Khmer-document set |
| Text and image inputs complemented each other | Hybrid answer correctness: 0.7580 vs. 0.7295 | Multimodal context can improve answer quality, with a latency trade-off |
| Table extraction remained incomplete | 74.70% global missing-cell rate for Surya OCR 2 | Matched-cell accuracy alone does not measure full structural recovery |
| A specialized detector was effective | YOLO26s mAP@0.5: 98.53% | Task-specific models can complement general-purpose document parsers |

**Conclusion.** This internship strengthened my experience in applied AI research, model evaluation, retrieval-augmented generation, and dataset construction for a low-resource language. It also showed that improving Khmer document intelligence requires evaluating the entire pipeline—from extraction and structure preservation to retrieval and answer generation—rather than relying on a single model metric.

---

## Report and attribution

This portfolio case study summarizes work carried out during my AI Research Internship at **Techo Startup Center, Cambodia**, from 1 June to 31 August 2026, under the supervision of **Dr. Rina Bouy**. The figures reported here are drawn from my internship report. Results correspond to the specific datasets and experimental settings described above and should not be interpreted as universal performance guarantees.