# PrivFed-GenMed
## A Privacy-Preserving Federated Generative AI Framework for Responsible Medical Text Generation

---

## Overview

PrivFed-GenMed is a privacy-preserving federated generative AI framework designed for medical text generation in regulated healthcare environments. The framework enables multiple medical institutions to collaboratively fine-tune a generative language model (GPT-2) without sharing raw patient data. Instead of centralizing sensitive clinical records, each institution performs local model training and shares only model updates through a federated learning pipeline coordinated using the Flower framework.

This project demonstrates the feasibility of integrating federated learning with generative language models to support privacy-sensitive clinical documentation tasks while maintaining competitive linguistic and semantic quality.

---

## Motivation

Large Language Models (LLMs) such as GPT-2 have shown strong performance in clinical text generation tasks, including medical note creation, summaries, and documentation support. However, traditional centralized training approaches require direct access to large volumes of electronic health records (EHRs), which violates strict privacy regulations such as HIPAA and GDPR.

Hospitals and medical institutions are therefore unable to participate in shared training pipelines due to confidentiality concerns. Federated Learning (FL) offers a decentralized alternative by allowing collaborative training without transferring raw data. While FL has been widely explored for classification and diagnostic tasks, its application to generative medical language models remains limited.

PrivFed-GenMed addresses this gap by applying federated learning to generative medical NLP, enabling responsible and privacy-preserving medical text generation.

---

## Key Contributions

- Federated fine-tuning of a GPT-2 language model for medical text generation  
- Privacy-preserving decentralized training using the Flower federated learning framework  
- Simulation of multiple hospital clients with isolated local datasets  
- Quantitative evaluation using training loss, perplexity, and BLEU score  
- Qualitative assessment of generated clinical narratives for coherence and realism  
- Demonstration that federated generative training does not compromise performance compared to centralized baselines  

---

## System Architecture

The PrivFed-GenMed framework follows a standard federated learning workflow adapted for generative models.

- Multiple simulated hospital clients act as independent data silos  
- Each client locally fine-tunes a GPT-2 model on synthetic medical text  
- No raw data or patient records are shared between clients or the server  
- A central federated server aggregates model updates using secure parameter averaging  
- The global model is redistributed to clients for subsequent training rounds  

This architecture ensures strict data privacy while enabling collaborative learning.

---

## Experimental Setup

- **Model:** GPT-2  
- **Frameworks:** PyTorch, Hugging Face Transformers, Flower  
- **Clients:** 3 simulated hospital clients  
- **Dataset:** Synthetic medical notes  
- **Federated Rounds:** 3  
- **Evaluation Metrics:** Training loss, perplexity, BLEU score, qualitative analysis  

A centralized baseline model was trained for comparison to evaluate the impact of federated fine-tuning on generative performance.

---

## Results Summary

Experimental results demonstrate that the federated global model converges more consistently than individual client models across training rounds.

- Progressive reduction in training loss  
- Lower perplexity compared to the centralized baseline  
- Improved BLEU scores indicating stronger linguistic quality  
- Medically coherent and contextually relevant generated text  

These findings confirm the feasibility of federated generative learning for medical NLP applications.

---

