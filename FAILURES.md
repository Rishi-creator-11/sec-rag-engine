# RAG Failures

## Failure 1 — Unsupported answer citation

Question:
What will NVIDIA's exact stock price be on December 31, 2027?

What happened:
The system correctly refused because the SEC excerpts cannot determine a future stock price, but it still attached [Source 1] [Source 4].

Expected:
Return the insufficient-evidence refusal without citations.

Failure type:
Citation / generation formatting issue.


## Failure 2 — Unsupported answer citation

Question:
What will Apple's revenue be in 2030?

What happened:
The system correctly refused because the filing does not provide an exact 2030 revenue figure, but it still attached [Source 1] [Source 3].

Expected:
Return the insufficient-evidence refusal without citations.

Failure type:
Citation / generation formatting issue.


## Failure 3 — Comparative retrieval coverage

Questions:
Which company discusses AI-related risks most directly?
Which company appears most exposed to China-related restrictions?

What happened:
The answers are reasonable, but top-k retrieval does not guarantee evidence from every company being compared.

Expected:
For cross-company comparison questions, retrieve relevant evidence independently for Apple, Microsoft, and NVIDIA before asking the LLM to compare them.

Failure type:
Retrieval design limitation.