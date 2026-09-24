# Ashik Hegde Context Graph & Indian Semantic Dataset (`mydata.md`)

This document details the semantic Indian banking and fintech dataset created for **Ashik Hegde** in Neo4j, explaining the graph architecture, decision traces, causal chains, and the **real showcaseable questions** you can ask the Context Graph AI platform to demonstrate its capabilities.

---

## 1. What Was Created & Seeded

A realistic, interconnected semantic graph dataset was generated and inserted directly into your live Neo4j database using [`backend/scripts/seed_ashik_data.py`](file:///media/ashik-hegde/Drive2/5edi/context-graph-demo/backend/scripts/seed_ashik_data.py).

```mermaid
graph TD
    Ashik["Person: Ashik Hegde<br/>(Bengaluru, CIBIL 795, Risk 0.08)"]
    Company["Organization: 5edi Technologies Pvt Ltd<br/>(AI & DeepTech Software)"]
    HDFC["Account: HDFC Salary Checking<br/>(₹8,45,000 / HDFC501004928172)"]
    Zerodha["Account: Zerodha Demat Trading<br/>(₹18,60,000 / ZR12903847)"]
    ICICI["Account: ICICI Forex EEFC<br/>(₹12,30,000 / ICIC0009482711)"]
    
    Dec1["Decision: DEC-ASHIK-2024-001<br/>(FEMA $12.5k Forex Remittance Approval)"]
    Dec2["Decision: DEC-ASHIK-2024-002<br/>(₹15 Lakh Prime Credit Limit Approval)"]
    Dec3["Decision: DEC-ASHIK-2024-003<br/>(SEBI ₹35 Lakh MTF Leverage Exception)"]
    Dec4["Decision: DEC-ASHIK-2024-004<br/>(UPI Rapid Velocity Alert Clearance)"]

    Ashik -->|WORKS_FOR (Founder)| Company
    Ashik -->|OWNS| HDFC
    Ashik -->|OWNS| Zerodha
    Ashik -->|OWNS| ICICI
    
    Dec1 -->|ABOUT| Ashik
    Dec2 -->|ABOUT| Ashik
    Dec3 -->|ABOUT| Ashik
    Dec4 -->|ABOUT| Ashik

    Dec1 -->|INFLUENCED (Weight: 0.88)| Dec2
    Dec2 -->|PRECEDENT_FOR (Relevance: 0.89)| Dec3
    Dec1 -->|CAUSED (Enabling: 0.85)| Dec3
```

---

## 2. Graph Node & Relationship Breakdown

### A. Core Entity: `Person`
* **Name**: Ashik Hegde
* **Canonical ID**: `canonical-ashik-hegde` (ID: `person-ashik-hegde-001`)
* **Location**: Bengaluru, Karnataka, India
* **Role**: Founder & Lead AI Systems Engineer at **5edi Technologies Private Limited**
* **KYC & Verification**: Verified Aadhaar + PAN (`ABCDE1234F`), CIBIL Score: **795** (Tier-1 Prime)
* **Risk Score**: `0.08` (Low Risk)
* **Source Systems**: `['HDFC_CoreBanking', 'Zerodha_Broking', '5edi_MCA_India', 'Credit_Bureau_CIBIL']`

### B. Organizations (`Organization`)
1. **5edi Technologies Private Limited** (`org-5edi-tech-001`): AI & DeepTech startup in Bengaluru (MCA Registration `U72900KA2022PTC158941`, Risk Rating: `A`, Sanctions: `clear`).
2. **HDFC Bank Limited** (`org-hdfc-bank-001`): Core banking and salary disbursements.
3. **Zerodha Broking Limited** (`org-zerodha-001`): Stock brokerage, Demat depository, and margin trading.
4. **ICICI Bank Limited** (`org-icici-bank-001`): Multi-currency forex remittances and trade services.
5. **Vertex Global AI Labs Inc** (`org-vertex-ai-001`): US-based enterprise client in San Francisco, originator of SWIFT software export payments.
6. **Razorpay Software Private Limited** (`org-razorpay-001`): Indian payment gateway counterparty for cloud server hosting.

### C. Financial Accounts (`Account`)
* **`ACC-HDFC-ASHIK-01`**: Checking/Salary account (`HDFC501004928172`), Balance ₹8,45,000 INR (Koramangala 4th Block Branch).
* **`ACC-ZERODHA-ASHIK-02`**: Demat Trading Margin account (`ZR12903847`), Balance ₹18,60,000 INR (Zerodha HQ Bengaluru).
* **`ACC-ICICI-ASHIK-03`**: EEFC Multi-currency Forex account (`ICIC0009482711`), Balance ₹12,30,000 INR (Indiranagar Branch).

### D. Transactions (`Transaction`)
* **`TXN-ASHIK-IN-001`** ($12,500 USD / ~₹10,40,000 INR): Inward SWIFT wire for AI consulting services from Vertex Global AI Labs Inc.
* **`TXN-ASHIK-IN-002`** (₹2,50,000 INR): Monthly NEFT salary credit from 5edi Technologies Pvt Ltd.
* **`TXN-ASHIK-IN-003`** (₹4,50,000 INR): Long equity investment into Nifty 50 Index & IT ETF via Zerodha Kite.
* **`TXN-ASHIK-IN-004`** (₹95,000 INR): Rapid UPI payment to Razorpay for AI GPU cloud infrastructure (Triggered temporary velocity alert).
* **`TXN-ASHIK-IN-005`** (₹1,20,000 INR): Outbound UPI lease payment for Koramangala Tech Park dedicated workspace.
* **`TXN-ASHIK-IN-006`** (₹3,00,000 INR): IMPS transfer from HDFC to Zerodha trading collateral pool.

### E. Regulatory Policies (`Policy`)
* **`pol-rbi-fema-001`**: *RBI FEMA Master Direction on Software Export & Forex Remittance* (Purpose code P0802 verification, Form A2/SOFTEX compliance).
* **`pol-credit-tier1-001`**: *Tier-1 Commercial Credit & Prime Credit Card Underwriting Policy* (CIBIL score ≥ 775, DTI < 35%, verified business cash flow).
* **`pol-sebi-mtf-001`**: *SEBI Margin Trading Facility & F&O Risk Containment Policy* (150%+ liquid bluechip collateral coverage, zero margin defaults).
* **`pol-upi-velocity-001`**: *NPCI & RBI High-Velocity Instant Payment (UPI) Monitoring Framework* (Device hardware binding, MCC validation).

### F. Decision Traces (`Decision`) — The "Event Clock"
Each decision contains **full rationale text**, risk factors, confidence scores, and **768-dimensional Gemini vector embeddings** (`gemini-embedding-001`):

1. **`DEC-ASHIK-2024-001` (Compliance Approval)**: Approved $12,500 inward forex software export wire under RBI FEMA guidelines after validating Purpose Code P0802 and clear OFAC screening. (Officer: *Priya Sharma*).
2. **`DEC-ASHIK-2024-002` (Credit Approval)**: Approved ₹15,00,000 unsecured prime credit limit & Visa Infinite card citing 795 CIBIL score, 14.2% DTI, and verified recurring foreign revenue from `DEC-ASHIK-2024-001`. (Officer: *Vikram Malhotra*).
3. **`DEC-ASHIK-2024-003` (Trading Exception Approval)**: Approved SEBI MTF intraday leverage limit increase from ₹15L to ₹35L backed by 210% collateral coverage in Nifty 50 holdings. Precedent established by `DEC-ASHIK-2024-002`. (Officer: *Rohit Verma*).
4. **`DEC-ASHIK-2024-004` (Fraud Velocity Review)**: Cleared rapid UPI burst alert `ALT-IN-UPI-8821` as false-positive legitimate B2B tech cloud vendor payouts authenticated by biometric 2FA. (Officer: *Vikram Malhotra*).

### G. Causal Graph Relationships
* `(DEC-ASHIK-2024-001)-[:INFLUENCED {weight: 0.88, type: "context"}]->(DEC-ASHIK-2024-002)`
* `(DEC-ASHIK-2024-002)-[:PRECEDENT_FOR {similarity_score: 0.91, outcome_relevance: 0.89}]->(DEC-ASHIK-2024-003)`
* `(DEC-ASHIK-2024-001)-[:CAUSED {confidence: 0.85, type: "enabling"}]->(DEC-ASHIK-2024-003)`

---

## 3. Real Showcaseable Questions to Ask the Platform

Open the Context Graph web application chat interface (`http://localhost:3000`) and test the following queries to demonstrate the full power of the system:

### 🌟 Category 1: Comprehensive Entity & Graph Overview
> **Prompt to ask:**
> ```text
> Show me everything about Ashik Hegde - his profile, linked accounts, organizations, and all historical decision traces made about him.
> ```
> **What this showcases:**
> * The agent calls `search_customer("Ashik Hegde")` and `get_graph_data_for_entity`.
> * Automatically populates the interactive **Context Graph Visualization** with Ashik Hegde, HDFC, Zerodha, ICICI accounts, and 5edi Technologies.
> * Displays the **Decision Trace Panel** listing all 4 decisions chronologically.

---

### 🌟 Category 2: Semantic Precedent Search & Credit Decisioning
> **Prompt to ask:**
> ```text
> A tech startup founder in Bangalore wants an unsecured credit limit increase to ₹15 Lakhs with a CIBIL score near 800 and foreign export revenue. Find similar past credit decisions and check relevant underwriting policies.
> ```
> **What this showcases:**
> * The agent uses `find_precedents` with **768-dimensional Gemini vector embeddings** to retrieve `DEC-ASHIK-2024-002`.
> * Retrieves the *Tier-1 Commercial Credit Underwriting Policy*.
> * Recommends approval and cites Ashik Hegde's past case as a direct precedent with similarity score > 0.85.

---

### 🌟 Category 3: Causal Chain Tracing ("Why was this approved?")
> **Prompt to ask:**
> ```text
> Trace the causal chain for Ashik Hegde's trading margin exception (DEC-ASHIK-2024-003). What past decisions and income events led to this approval?
> ```
> **What this showcases:**
> * The agent calls `get_causal_chain("DEC-ASHIK-2024-003", direction="upstream")`.
> * Traverses the graph: `DEC-ASHIK-2024-001 (Forex Inflow)` ➔ `DEC-ASHIK-2024-002 (Prime Credit Approval)` ➔ `DEC-ASHIK-2024-003 (MTF Leverage Exception)`.
> * Proves the "Event Clock" concept: reasoning why liquidity from software exports enabled credit rating, which in turn enabled collateralized trading leverage.

---

### 🌟 Category 4: Regulatory Policy & FEMA Forex Compliance Check
> **Prompt to ask:**
> ```text
> What policies apply when receiving a $12,500 foreign inward wire for software consulting services under RBI FEMA rules? Were there any past approvals for Ashik Hegde under this policy?
> ```
> **What this showcases:**
> * The agent triggers `get_policy("compliance", policy_name="FEMA")`.
> * Retrieves Purpose Code P0802 requirements and Form A2 / SOFTEX documentation rules.
> * Links directly to `DEC-ASHIK-2024-001` and compliance officer *Priya Sharma*.

---

### 🌟 Category 5: Fraud & Velocity Alert Investigation
> **Prompt to ask:**
> ```text
> Analyze account ACC-HDFC-ASHIK-01 for any recent flagged transactions or velocity alerts. What was the investigation outcome?
> ```
> **What this showcases:**
> * Finds flagged transaction `TXN-ASHIK-IN-004` and resolved alert `ALT-IN-UPI-8821`.
> * Explains the resolution in `DEC-ASHIK-2024-004` (device biometric verification and legitimate Razorpay B2B cloud invoice).

---

## 4. Useful Cypher Queries for Neo4j Browser

If you want to view the raw graph directly inside the **Neo4j Browser** (`http://localhost:7474`):

#### 1. View Ashik Hegde's complete subgraph:
```cypher
MATCH (p:Person {name: 'Ashik Hegde'})-[r1]-(n)
OPTIONAL MATCH (n)-[r2]-(m)
RETURN p, r1, n, r2, m
LIMIT 50;
```

#### 2. View all Decision Traces & Causal Chains:
```cypher
MATCH (d1:Decision)-[r:INFLUENCED|PRECEDENT_FOR|CAUSED]->(d2:Decision)
RETURN d1, r, d2;
```

#### 3. View Policies linked to Decisions:
```cypher
MATCH (d:Decision)-[:APPLIED_POLICY]->(pol:Policy)
MATCH (d)-[:MADE_BY]->(e:Employee)
RETURN d.reasoning_summary, pol.name, e.name, d.confidence_score;
```

---

## 5. How to Re-Run or Extend the Script

The script is located at [`backend/scripts/seed_ashik_data.py`](file:///media/ashik-hegde/Drive2/5edi/context-graph-demo/backend/scripts/seed_ashik_data.py).

To re-seed or update at any time:
```bash
# From workspace root
backend/.venv/bin/python backend/scripts/seed_ashik_data.py
```
It runs idempotently (cleans up any previous Ashik Hegde records before creating fresh, consistent nodes with updated Gemini vector embeddings).

