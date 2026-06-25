\# Vietnamese Student HackAI 2026 - Board C



\## Overview



This project is an Three-Agent Verification Pipeline system for multiple-choice question answering.



The system uses a three-agent verification pipeline:



\* \*\*Solver\*\*: Solves the question and selects the best answer.

\* \*\*Critic\*\*: Reviews the Solver's reasoning and only challenges it when there is concrete evidence.

\* \*\*Judge\*\*: Invoked only when Solver and Critic disagree to make the final decision.



This design reduces unnecessary model calls while maintaining answer quality.



\---



\## Project Structure



```text

run.py              Main program

Dockerfile          Docker configuration

requirements.txt    Python dependencies

prompt.txt          Prompt template

Agent.txt           Agent architecture

```



\---



\## Build Docker



```bash

docker build -t hackai .

```



\---



\## Run Docker



```bash

docker run --rm \\

\-v /data:/data \\

\-v /output:/output \\

hackai

```



\---



\## Input



The program automatically detects:



\* `/data/private\_test.json`

\* `/data/public\_test.json`



If neither exists, it falls back to:



```text

public\_test.json

```



for local testing.



\---



\## Output



The program generates:



```text

/output/pred.csv

```



or



```text

result.csv

```



when running locally.



CSV format:



```text

qid,answer

test\_0001,A

test\_0002,C

...

```



\---



\## Pipeline



```text

Question

&#x20;   │

&#x20;   ▼

Solver

&#x20;   │

&#x20;   ▼

Critic

&#x20;   │

&#x20;┌──┴──┐

&#x20;│     │

Agree Disagree

&#x20;│      │

&#x20;▼      ▼

Final  Judge

&#x20;         │

&#x20;         ▼

&#x20;   Final Answer

```



\---



\## Requirements



\* Python 3.11

\* Google GenAI SDK

\* Docker



