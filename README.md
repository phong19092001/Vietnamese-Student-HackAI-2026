# Vietnamese Student HackAI 2026 - Board C

## Overview

This project implements a **Three-Agent Verification Pipeline** for multiple-choice question answering.

The system consists of three collaborative agents:

* **Solver**: analyzes the question and generates an initial answer.
* **Critic**: reviews the Solver's reasoning and only challenges it when there is concrete evidence.
* **Judge**: is invoked **only when Solver and Critic disagree** to determine the final answer.

The project runs completely **offline** using a locally deployed **Qwen2.5-0.5B-Instruct** model and does not require Internet access during inference.

---

## Project Structure

```text
.
├── run.py
├── Dockerfile
├── requirements.txt
├── README.md
└── models/
    └── models/
(optional, packaged inside Docker image)
```

---

## Build Docker

```bash
docker build -t hackaithon2026 .
```

---

## Run Docker

The competition organizer mounts the test file to:

```text
/code/private_test.json
```

Run:

```bash
docker run --rm hackaithon2026
```

If additional Docker flags (such as `--gpus all` or `--ipc=host`) are required by the organizer, they can be added when launching the container.

---

## Input

The program automatically detects:

```text
/code/private_test.json
```

For local testing it also supports:

```text
private_test.json
public_test.json
```

---

## Output

The program generates two output files in the working directory:

```text
submission.csv
submission_time.csv
```

Example:

```text
submission.csv

qid,answer
test_0001,A
test_0002,C
```

```text
submission_time.csv

qid,time
test_0001,0.842315
test_0002,0.771204
```

---

## Pipeline

```text
Question
   │
   ▼
Solver
   │
   ▼
Critic
   │
┌──┴──┐
│     │
Agree Disagree
│      │
▼      ▼
Final  Judge
         │
         ▼
   Final Answer
```

---

## Model

* Qwen2.5-0.5B-Instruct
* Local inference
* Offline execution
* No Internet connection required

---

## Requirements

* Python 3.11
* PyTorch
* Transformers
* Accelerate
* SentencePiece
* Safetensors
* Hugging Face Hub

---

## Notes

* The submitted Docker image already contains the required model for offline inference.
* Judge is only executed when Solver and Critic disagree, reducing unnecessary inference while maintaining answer quality.
* The program outputs both prediction results and per-question inference time as required by the competition.
