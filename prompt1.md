You are helping construct a question-driven multimodal summarisation dataset. Your task is to generate 2–5 high-quality triplets that include:

- A user question that requires multimodal summarisation(e.g., layout, content, tables, figures).
- One or more intent labels.
- Up to 10 paragraph-level evidence snippets.

**Intent Types (multiple label):**
- Descriptive: clarify definitions, problem settings, or describe an architecture/component
- Procedural: explain methods, algorithms, or step-by-step processes
- Causal: discuss cause-effect relations  
- Verificative: validate claims or assumptions through evidence  
- Comparative: compare methods, baselines, settings, or results
- Evaluative: judge effectiveness or limitations  

For each QA pair, follow this format:
```json
{
  "question": "...",
  "intent": ["Descriptive", "Procedural", ...], 
  "evidence": [{
    "section": "X.X Introduction", 
    "type": "paragraph",
    "content": "Starts with ‘…’",
    },],
}
```
**Output Format :JSON only**: A list of 2-5 entries: [{},{},...]

Constraints:

**Reasoning Across Paragraphs**: Questions should encourage reasoning across multiple paragraphs when possible.

**Reference Requirement**: In evidence, the "section" field should specify the full subsection title using its original label in the document. The "content" field should only include sentences from the input document.
