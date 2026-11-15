<div align="center">

# 🚀 AlgoX — Mars Colony Emergency Response Optimizer

**Autonomous Multi-Crew Incident Scheduling Optimization Engine & Mission Control Dashboard**

[![Live Demo](https://img.shields.io/badge/Live_Demo-Vercel-black?style=for-the-badge&logo=vercel)](https://algo-x-iota.vercel.app)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

[**Explore Live Application →**](https://algo-x-iota.vercel.app)

</div>

---

## 📖 Executive Summary

**AlgoX** is a high-performance optimization system engineered to solve the **Mars Colony Emergency Response Scheduling Problem**. In an isolated Martian habitat, emergencies arise across three specialized operational domains:

- 🔴 **Structural (`E`):** Hull fissures, atmospheric containment breaches, corridor integrity failures.
- 🔵 **Medical (`M`):** Crew injuries, radiation sickness, emergency life stabilization.
- 🟣 **Life Support (`L`):** Atmospheric scrubber malfunction, power grid routing, hydrological recycling.

Every incident carries an arrival timestamp, operational category, severity level ($1 \dots 10$), and baseline duration. The core challenge is assigning and scheduling all incidents across the three specialized crews to **maximize the cumulative Colony Safety Score** under severe time-decay penalties and strict physical non-overlap constraints.

---

## 🧮 Mathematical Problem Formulation

### 1. Dynamic Urgency Window
Incidents with higher severity deteriorate faster. The acceptable response window $\mathcal{W}_i$ decays inversely with severity:
$$\mathcal{W}_i = 60 - 4 \times \text{severity}_i \quad (\text{minutes})$$

### 2. Time-Decayed Safety Score
If response begins after $\text{arrival}_i + \mathcal{W}_i$, the incident yields zero points. Within the window, score contribution decays linearly with wait time:
$$\text{Score}_i = \text{severity}_i \times \max\left(0, \; 1 - \frac{\text{start}_i - \text{arrival}_i}{\mathcal{W}_i}\right)$$

### 3. Cross-Crew Assistance Penalty
Any crew can assist with any incident, but non-matching assignments double the execution duration:
$$\text{duration}_i(\text{crew}) = \begin{cases} \text{dur}_i & \text{if } \text{crew} = \text{category}_i \\ 2 \times \text{dur}_i & \text{if } \text{crew} \neq \text{category}_i \end{cases}$$

### 4. Objective Function
$$\max \sum_{i \in \mathcal{I}} \text{Score}_i \quad \text{subject to:}$$
$$\begin{aligned}
\text{start}_i &\ge \text{arrival}_i && \forall i \in \mathcal{I} \\
\text{end}_i &= \text{start}_i + \text{duration}_i(\text{crew}_i) && \forall i \in \mathcal{I} \\
[\text{start}_j, \text{end}_j) \cap [\text{start}_k, \text{end}_k) &= \emptyset && \forall j \neq k \text{ where } \text{crew}_j = \text{crew}_k
\end{aligned}$$

---

## ⚡ Algorithm Architecture

Our solution combines a **60-configuration deterministic greedy grid** with an **asynchronous 4-process parallel Simulated Annealing metaheuristic**:

```
                  ┌────────────────────────────────────────┐
                  │          Incident CSV Log              │
                  └───────────────────┬────────────────────┘
                                      ▼
                  ┌────────────────────────────────────────┐
                  │   60-Config Greedy Initialization      │
                  │ (Multi-parameter composite urgency grid│
                  └───────────────────┬────────────────────┘
                                      ▼
                  ┌────────────────────────────────────────┐
                  │       Active / Dropped Partition       │
                  │  Active Queue: High potential score    │
                  │  Dropped Pool: Zero-score candidates   │
                  └───────────────────┬────────────────────┘
                                      ▼
           ┌──────────────────────────┴──────────────────────────┐
           │        Parallel Multi-Process Annealing (4 Cores)   │
           │  Worker 1: (T0=2.0, α=0.01)   Worker 2: (T0=1.0, α=0.01) │
           │  Worker 3: (T0=4.0, α=0.005)  Worker 4: (T0=2.0, α=0.003)│
           │                                                     │
           │  7 Neighborhood Operators + Early-Exit Pruning      │
           └──────────────────────────┬──────────────────────────┘
                                      ▼
                  ┌────────────────────────────────────────┐
                  │       Schedule Reconstruction          │
                  │ (Pack active queue -> non-interfering   │
                  │  greedy insertion of dropped incidents)│
                  └───────────────────┬────────────────────┘
                                      ▼
                  ┌────────────────────────────────────────┐
                  │       7-Rule Strict Judge Audit        │
                  │        Output: submission.json         │
                  └────────────────────────────────────────┘
```

### Key Technical Innovations:
1. **Active vs. Dropped Incident Splitting:**
   Separating incidents that mathematically cannot score points into a secondary dropped pool reduces the combinatorial state space by $\sim 30\%$, enabling the search space to focus exclusively on score-generating jobs.
2. **7 Neighborhood Move Operators:**
   - `SHIFT`: Intra-queue displacement.
   - `SWAP`: Intra-queue job swap.
   - `TRANSFER`: Inter-queue single-job migration.
   - `EXCHANGE`: Inter-queue pairwise job swap.
   - `BLOCK_REVERSE`: Inversion of contiguous schedule slices.
   - `DROP_SWAP`: Exchange between active job and zero-point dropped job.
   - `DROP_INSERT`: Opportunistic insertion of dropped incident into idle slack time.
3. **$O(1)$ Early-Exit Incremental Delta Evaluation:**
   Maintains cached timeline boundaries (`endT[]`, `cum[]`) to prune unviable mutations without recomputing the entire schedule.
4. **Deterministic Reproducibility:**
   Fixed seed propagation across all random generators guarantees $100\%$ bitwise reproducible schedules.

---

## 📊 Benchmark & Score Results

Evaluated across standard benchmarks (1,000,000 iterations, Seed: `2026`):

| Dataset | Total Incidents | Greedy Baseline | AlgoX Final Score | Optimization Gain | Verification |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Sample 1** | 400 | 1450.20 | **1493.74** | +43.54 pts | `valid` ✅ |
| **Sample 2** | 400 | 1089.45 | **1124.67** | +35.22 pts | `valid` ✅ |
| **Sample 3** | 400 | 1332.18 | **1370.77** | +38.59 pts | `valid` ✅ |
| **Stress Test 1** | 4,000 | 11,840.10 | **12,355.20** | +515.10 pts | `valid` ✅ |
| **Stress Test 2** | 10,000 | 29,210.00 | **30,501.40** | +1,291.40 pts | `valid` ✅ |

*All results verified using `judge.py` checking all 7 official validity constraints.*

---

## 🖥️ Mission Control Web UI

The repository includes a modern, responsive web application built with **Flask**, **Vanilla JavaScript (ES6+)**, and **HTML5 Canvas**:

- **Real-Time Canvas Gantt Chart:** 60 FPS interactive crew dispatch timeline supporting mouse-drag panning, zoom scaling, and per-incident hover tooltips.
- **Interactive Score Calculator:** Visual decay-curve simulator demonstrating real-time score degradation.
- **Telemetry & Severity Analytics:** Distribution metrics comparing points captured versus theoretical maximum ceiling.
- **Searchable Schedule Data Table:** Paginated, multi-column sortable table with severity pills and crew badges.
- **Zero-Latency In-Browser Export:** Instant `submission.json` generation and clipboard copy.

---

## 📁 Repository Structure

```
AlgoX/
├── data/                                  # Benchmark incident logs
│   ├── sample1.csv                        # 400 incidents (mixed categories)
│   ├── sample2.csv                        # 400 incidents (high-severity distribution)
│   └── sample3.csv                        # 400 incidents (burst arrival distribution)
├── docs/                                  # Problem specification & rules
│   ├── AlgoX Problem Statement.pdf        # Formal competition problem brief
│   └── AlgoX Rules, Evaluation and Marks.pdf # Evaluation rubric & validation rules
├── templates/
│   └── index.html                         # Interactive mission control dashboard UI
├── app.py                                 # Flask web application & REST APIs
├── judge.py                               # Official constraint validation engine
├── solve.py                               # Core metaheuristic solver & parallel annealer
├── requirements.txt                       # Minimal dependencies (Flask)
├── vercel.json                            # Vercel serverless deployment config
└── README.md                              # Technical documentation
```

---

## 🛠️ Local Installation & Usage

### 1. Clone Repository
```bash
git clone https://github.com/mohands0810-debug/AlgoX.git
cd AlgoX
```

### 2. Environment Setup
```bash
pip install -r requirements.txt
```

### 3. Run via CLI
Execute the solver on any dataset:
```bash
python solve.py data/sample1.csv submission.json 1000000 2026
```

Validate schedule compliance:
```bash
python judge.py data/sample1.csv submission.json
```

### 4. Run Mission Control Web UI
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

---

## 🌐 Cloud Deployment

AlgoX is configured for instant serverless deployment on **Vercel**:
- Builder: `@vercel/python`
- Stateless synchronous execution optimized for serverless execution timeouts
- Live URL: **[https://algo-x-iota.vercel.app](https://algo-x-iota.vercel.app)**

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
