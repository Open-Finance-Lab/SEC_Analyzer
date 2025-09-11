# SEC Analyzer: Evaluating LLMs' Reasoning Failures on SEC Filings
**Abstract:** Analyzing SEC filings is a highly professional task in business and finance. However, existing general-purpose large language models (LLMs) have poor performance on analyzing SEC filings, e.g., GPT-4-Turbo (closed-book setting) only correctly answered $9\%$ of the questions in the FinanceBench dataset (curated by human experts). In this paper, we review $2400$ model answers on $150$ questions and identify typical reasoning failures. There were three LLMs (i.e., Claude2 \cite{CARUCCIO2024200336}, GPT-4 \cite{achiam2023gpt}, Llama2 \cite{touvron2023llama}) tested under 6 settings, including closed book, oracle, single vector store, shared vector store, long context and reverse prompt, with a total of 16 model configurations \cite{islam2023financebench}.   
  There are six major reasoning failures: information extraction error, misuse of concepts, miscalculation of financial ratios, comparison errors of values, misunderstanding of context, and wrong unit. Our findings highlight the reasoning challenges current LLMs still face when analyzing SEC filings, challenges that may directly affect valuation, compliance assessments, risk management, etc. 

---

# Reference
This project uses the open-source dataset and SEC filings from the [FinanceBench](https://github.com/patronus-ai/financebench).

```bibtex
@misc{islam2023financebench,
      title={FinanceBench: A New Benchmark for Financial Question Answering}, 
      author={Pranab Islam and Anand Kannappan and Douwe Kiela and Rebecca Qian and Nino Scherrer and Bertie Vidgen},
      year={2023},
      eprint={2311.11944},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
