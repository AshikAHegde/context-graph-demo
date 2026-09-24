#!/usr/bin/env python3
"""
Seed realistic Indian semantic data centered around Ashik Hegde into Neo4j.
Includes unified entity profile, Indian banking/tech accounts, transactions,
policies (RBI FEMA, SEBI, NPCI UPI, Credit Underwriting), decision traces,
causal chains, alerts, and support tickets with 768-dim Gemini embeddings.
"""

import os
import random
from datetime import datetime
from dotenv import load_dotenv
from neo4j import GraphDatabase
from google import genai
from google.genai import types

load_dotenv()

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")


class AshikDataSeeder:
    def __init__(self):
        print(f"Connecting to Neo4j database: {NEO4J_DATABASE} at {NEO4J_URI}...")
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
        self.gemini_client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None
        self.embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
        self.embedding_dimensions = int(os.getenv("GEMINI_EMBEDDING_DIMENSIONS", "768"))

    def close(self):
        self.driver.close()

    def get_embedding(self, text: str) -> list[float]:
        """Generate 768-dim Gemini text embedding with fallback."""
        if self.gemini_client:
            try:
                config_params = types.EmbedContentConfig(output_dimensionality=self.embedding_dimensions)
                resp = self.gemini_client.models.embed_content(
                    model=self.embedding_model,
                    contents=text,
                    config=config_params,
                )
                return resp.embeddings[0].values
            except Exception as e:
                print(f"Warning: Failed to generate Gemini embedding ({e}), using normalized pseudo-vector.")
        
        # Fallback reproducible deterministic vector if API is unreachable
        random.seed(abs(hash(text)) % (2**32))
        vec = [random.gauss(0, 1) for _ in range(self.embedding_dimensions)]
        norm = sum(x**2 for x in vec) ** 0.5
        return [x / norm for x in vec]

    def get_fastrp_dummy(self) -> list[float]:
        """Generate 128-dim FastRP structural vector."""
        vec = [random.gauss(0, 1) for _ in range(128)]
        norm = sum(x**2 for x in vec) ** 0.5
        return [x / norm for x in vec]

    def cleanup_existing_ashik_data(self):
        """Remove any previous Ashik Hegde specific nodes for clean re-seeding."""
        print("Cleaning up any existing Ashik Hegde test records...")
        with self.driver.session(database=NEO4J_DATABASE) as session:
            session.run("""
                MATCH (n)
                WHERE n.id STARTS WITH 'person-ashik-hegde'
                   OR n.id STARTS WITH 'ACC-HDFC-ASHIK'
                   OR n.id STARTS WITH 'ACC-ZERODHA-ASHIK'
                   OR n.id STARTS WITH 'ACC-ICICI-ASHIK'
                   OR n.id STARTS WITH 'TXN-ASHIK'
                   OR n.id STARTS WITH 'DEC-ASHIK'
                   OR n.id STARTS WITH 'ALT-IN-UPI'
                   OR n.id STARTS WITH 'TKT-IN'
                   OR n.id STARTS WITH 'org-5edi-tech'
                   OR n.id STARTS WITH 'pol-rbi-fema'
                   OR n.id STARTS WITH 'pol-sebi-mtf'
                   OR n.id STARTS WITH 'pol-credit-tier1'
                   OR n.id STARTS WITH 'pol-upi-velocity'
                DETACH DELETE n
            """)
        print("Cleanup done.")

    def seed_all(self):
        self.cleanup_existing_ashik_data()

        with self.driver.session(database=NEO4J_DATABASE) as session:
            # 1. Create Ashik Hegde Person Node
            print("1. Creating Ashik Hegde Person node...")
            person_fastrp = self.get_fastrp_dummy()
            session.run("""
                MERGE (p:Person {id: 'person-ashik-hegde-001'})
                SET p.name = 'Ashik Hegde',
                    p.canonical_id = 'canonical-ashik-hegde',
                    p.normalized_name = 'ashik hegde',
                    p.email = 'ashik.hegde@5edi.com',
                    p.phone = '+91-9880194820',
                    p.date_of_birth = date('1998-05-14'),
                    p.city = 'Bengaluru',
                    p.state = 'Karnataka',
                    p.country = 'India',
                    p.pan = 'ABCDE1234F',
                    p.cibil_score = 795,
                    p.occupation = 'Founder & Lead AI Systems Engineer',
                    p.kyc_status = 'VERIFIED_AADHAAR_PAN',
                    p.risk_score = 0.08,
                    p.source_systems = ['HDFC_CoreBanking', 'Zerodha_Broking', '5edi_MCA_India', 'Credit_Bureau_CIBIL'],
                    p.fastrp_embedding = $fastrp_embedding,
                    p.created_at = datetime('2022-01-10T10:00:00Z'),
                    p.updated_at = datetime()
            """, {"fastrp_embedding": person_fastrp})

            # 2. Create Organizations
            print("2. Creating Organizations (5edi Technologies, HDFC, Zerodha, ICICI, Razorpay, Vertex AI)...")
            organizations = [
                {
                    "id": "org-5edi-tech-001",
                    "name": "5edi Technologies Private Limited",
                    "normalized_name": "5edi technologies private limited",
                    "type": "corporation",
                    "industry": "AI & DeepTech Software",
                    "country": "India",
                    "city": "Bengaluru",
                    "registration_no": "U72900KA2022PTC158941",
                    "risk_rating": "A",
                    "sanctions_status": "clear",
                    "source_systems": ["MCA_India", "GSTN_Karnataka", "HDFC_Corporate"],
                },
                {
                    "id": "org-hdfc-bank-001",
                    "name": "HDFC Bank Limited",
                    "normalized_name": "hdfc bank limited",
                    "type": "bank",
                    "industry": "Banking & Financial Services",
                    "country": "India",
                    "city": "Mumbai",
                    "risk_rating": "A",
                    "sanctions_status": "clear",
                    "source_systems": ["RBI_Regulated"],
                },
                {
                    "id": "org-zerodha-001",
                    "name": "Zerodha Broking Limited",
                    "normalized_name": "zerodha broking limited",
                    "type": "broker",
                    "industry": "Fintech & Stock Brokerage",
                    "country": "India",
                    "city": "Bengaluru",
                    "risk_rating": "A",
                    "sanctions_status": "clear",
                    "source_systems": ["SEBI_Regulated"],
                },
                {
                    "id": "org-icici-bank-001",
                    "name": "ICICI Bank Limited",
                    "normalized_name": "icici bank limited",
                    "type": "bank",
                    "industry": "Banking & Forex Services",
                    "country": "India",
                    "city": "Mumbai",
                    "risk_rating": "A",
                    "sanctions_status": "clear",
                    "source_systems": ["RBI_Regulated"],
                },
                {
                    "id": "org-razorpay-001",
                    "name": "Razorpay Software Private Limited",
                    "normalized_name": "razorpay software private limited",
                    "type": "vendor",
                    "industry": "Payment Gateway & Cloud Billing",
                    "country": "India",
                    "city": "Bengaluru",
                    "risk_rating": "A",
                    "sanctions_status": "clear",
                    "source_systems": ["NPCI_Partner"],
                },
                {
                    "id": "org-vertex-ai-001",
                    "name": "Vertex Global AI Labs Inc",
                    "normalized_name": "vertex global ai labs inc",
                    "type": "corporation",
                    "industry": "Enterprise AI & Cloud Systems",
                    "country": "United States",
                    "city": "San Francisco",
                    "risk_rating": "A",
                    "sanctions_status": "clear",
                    "source_systems": ["OFAC_Screened", "SWIFT_Network"],
                },
            ]

            for org in organizations:
                session.run("""
                    MERGE (o:Organization {id: $id})
                    SET o.name = $name,
                        o.normalized_name = $normalized_name,
                        o.type = $type,
                        o.industry = $industry,
                        o.country = $country,
                        o.city = $city,
                        o.risk_rating = $risk_rating,
                        o.sanctions_status = $sanctions_status,
                        o.source_systems = $source_systems,
                        o.created_at = datetime('2022-01-01T00:00:00Z'),
                        o.updated_at = datetime()
                """, org)

            # Link Ashik to 5edi Technologies
            session.run("""
                MATCH (p:Person {id: 'person-ashik-hegde-001'})
                MATCH (o:Organization {id: 'org-5edi-tech-001'})
                MERGE (p)-[r:WORKS_FOR]->(o)
                SET r.role = 'Founder & Lead AI Systems Engineer',
                    r.since = date('2022-01-15'),
                    r.equity_ownership_pct = 65.0,
                    r.verified = true
            """)

            # 3. Create Accounts
            print("3. Creating Indian Bank, Demat, and Forex Accounts...")
            accounts = [
                {
                    "id": "ACC-HDFC-ASHIK-01",
                    "account_number": "HDFC501004928172",
                    "account_type": "checking",
                    "status": "active",
                    "balance": 845000.0,
                    "currency": "INR",
                    "risk_tier": "low",
                    "opened_date": "2022-02-01",
                    "source_system": "HDFC_CoreBanking",
                    "bank_branch": "Koramangala 4th Block, Bengaluru",
                    "fastrp_embedding": self.get_fastrp_dummy(),
                },
                {
                    "id": "ACC-ZERODHA-ASHIK-02",
                    "account_number": "ZR12903847",
                    "account_type": "trading",
                    "status": "active",
                    "balance": 1860000.0,
                    "currency": "INR",
                    "risk_tier": "low",
                    "opened_date": "2022-03-15",
                    "source_system": "Zerodha_Kite_API",
                    "bank_branch": "Zerodha Broking Bengaluru HQ",
                    "fastrp_embedding": self.get_fastrp_dummy(),
                },
                {
                    "id": "ACC-ICICI-ASHIK-03",
                    "account_number": "ICIC0009482711",
                    "account_type": "savings",
                    "status": "active",
                    "balance": 1230000.0,
                    "currency": "INR",
                    "risk_tier": "low",
                    "opened_date": "2023-01-10",
                    "source_system": "ICICI_TradeForex",
                    "bank_branch": "Indiranagar, Bengaluru",
                    "fastrp_embedding": self.get_fastrp_dummy(),
                },
            ]

            for acc in accounts:
                session.run("""
                    MERGE (a:Account {id: $id})
                    SET a.account_number = $account_number,
                        a.account_type = $account_type,
                        a.status = $status,
                        a.balance = $balance,
                        a.currency = $currency,
                        a.risk_tier = $risk_tier,
                        a.opened_date = date($opened_date),
                        a.source_system = $source_system,
                        a.bank_branch = $bank_branch,
                        a.fastrp_embedding = $fastrp_embedding,
                        a.created_at = datetime('2022-02-01T09:00:00Z'),
                        a.updated_at = datetime()
                    WITH a
                    MATCH (p:Person {id: 'person-ashik-hegde-001'})
                    MERGE (p)-[:OWNS]->(a)
                """, acc)

            # Link 5edi corporate account relationship
            session.run("""
                MATCH (o:Organization {id: 'org-5edi-tech-001'})
                MATCH (a:Account {id: 'ACC-HDFC-ASHIK-01'})
                MERGE (o)-[:OWNS_ACCOUNT]->(a)
            """)

            # 4. Create Compliance / Risk Employees
            print("4. Creating Compliance & Risk Officers (Priya Sharma, Vikram Malhotra, Rohit Verma)...")
            employees = [
                {
                    "id": "emp-priya-sharma-001",
                    "employee_id": "EMP-IND-0042",
                    "name": "Priya Sharma",
                    "department": "Compliance",
                    "role": "Senior FEMA & AML Compliance Officer",
                    "authorization_level": 4,
                },
                {
                    "id": "emp-vikram-malhotra-001",
                    "employee_id": "EMP-IND-0089",
                    "name": "Vikram Malhotra",
                    "department": "Risk",
                    "role": "Senior Credit & Risk Desk Manager",
                    "authorization_level": 4,
                },
                {
                    "id": "emp-rohit-verma-001",
                    "employee_id": "EMP-IND-0115",
                    "name": "Rohit Verma",
                    "department": "Trading",
                    "role": "Margin & Derivatives Risk Desk Analyst",
                    "authorization_level": 3,
                },
            ]

            for emp in employees:
                session.run("""
                    MERGE (e:Employee {id: $id})
                    SET e.employee_id = $employee_id,
                        e.name = $name,
                        e.department = $department,
                        e.role = $role,
                        e.authorization_level = $authorization_level,
                        e.created_at = datetime('2021-06-01T09:00:00Z')
                """, emp)

            # 5. Create Indian Regulatory & Banking Policies
            print("5. Creating Regulatory Policies (RBI FEMA, SEBI MTF, Credit Underwriting, NPCI UPI)...")
            policies = [
                {
                    "id": "pol-rbi-fema-001",
                    "name": "RBI FEMA Master Direction on Software Export & Forex Remittance",
                    "category": "compliance",
                    "version": "v4.2",
                    "description": (
                        "RBI FEMA Master Direction guidelines for processing inward forex remittances exceeding $10,000 USD "
                        "for IT, software consultancy, and AI engineering services under Purpose Code P0802 with SOFTEX / Form A2 documentation. "
                        "Requires verified counterparty screening, invoice reconciliation, and AML clearance."
                    ),
                    "rules": [
                        "Inward remittances > $10,000 require Purpose Code P0802 or P0807 verification",
                        "Valid tax invoice and bilateral service agreement contract required",
                        "OFAC/FATF sanctions clearance mandatory for originating institution",
                    ],
                },
                {
                    "id": "pol-credit-tier1-001",
                    "name": "Tier-1 Commercial Credit & Prime Credit Card Underwriting Policy",
                    "category": "credit",
                    "version": "v3.0",
                    "description": (
                        "Underwriting criteria for approving unsecured premium credit limits above ₹10,00,000 (INR 1 Million) "
                        "for technology founders and executives. Evaluates CIBIL credit score (>= 775), debt-to-income (DTI < 35%), "
                        "and 12+ months verified software business/salary cash flows."
                    ),
                    "rules": [
                        "Minimum CIBIL bureau score threshold of 775",
                        "Maximum debt-to-income ratio capped at 35%",
                        "Proven cash flow runway and clean 24-month repayment history",
                    ],
                },
                {
                    "id": "pol-sebi-mtf-001",
                    "name": "SEBI Margin Trading Facility & F&O Risk Containment Policy",
                    "category": "trading",
                    "version": "v2.5",
                    "description": (
                        "SEBI rules and collateral haircut thresholds governing intraday margin exceptions and multi-asset "
                        "derivative exposure for accredited quantitative traders. Mandates 150%+ collateral coverage in approved bluechips."
                    ),
                    "rules": [
                        "Minimum collateral margin coverage of 150% in approved securities",
                        "Zero margin call defaults in trailing 12 months",
                        "Mandatory automated stop-loss risk controls enabled on algorithmic terminal",
                    ],
                },
                {
                    "id": "pol-upi-velocity-001",
                    "name": "NPCI & RBI High-Velocity Instant Payment (UPI) Monitoring Framework",
                    "category": "fraud",
                    "version": "v5.1",
                    "description": (
                        "Automated fraud velocity containment framework for rapid successive UPI transfers and corporate API payout batches. "
                        "Monitors geographic consistency, device hardware binding, and merchant category code (MCC) legitimacy."
                    ),
                    "rules": [
                        "Triggers alert if > 5 rapid transfers occur within 30 minutes totaling > ₹2,50,000",
                        "Requires SIM & device hardware binding check with Bengaluru IP geo-matching",
                        "Auto-clears if payee is a verified B2B GSTN-registered tech cloud vendor",
                    ],
                },
            ]

            for pol in policies:
                pol_emb = self.get_embedding(pol["name"] + ": " + pol["description"])
                session.run("""
                    MERGE (p:Policy {id: $id})
                    SET p.name = $name,
                        p.category = $category,
                        p.version = $version,
                        p.description = $description,
                        p.rules = $rules,
                        p.description_embedding = $embedding,
                        p.created_at = datetime('2022-01-01T00:00:00Z'),
                        p.updated_at = datetime()
                """, {**pol, "embedding": pol_emb})

            # 6. Create Transactions
            print("6. Creating Transactions (Forex SWIFT wire, Salary, NSE Demat trade, UPI vendor payouts)...")
            transactions = [
                {
                    "id": "TXN-ASHIK-IN-001",
                    "transaction_id": "TXN90481029384",
                    "type": "deposit",
                    "amount": 1040000.0,  # ~12,500 USD in INR
                    "currency": "INR",
                    "timestamp": "2024-04-16T14:20:00Z",
                    "status": "completed",
                    "channel": "wire",
                    "description": "Inward SWIFT Wire Remittance - Software AI Consulting Services Inv #5EDI-2024-04 from Vertex Global AI Labs Inc ($12,500 USD)",
                    "risk_score": 0.12,
                    "source_system": "ICICI_TradeForex",
                    "account_id": "ACC-ICICI-ASHIK-03",
                    "counterparty_org_id": "org-vertex-ai-001",
                    "fastrp_embedding": self.get_fastrp_dummy(),
                },
                {
                    "id": "TXN-ASHIK-IN-002",
                    "transaction_id": "TXN90481029385",
                    "type": "deposit",
                    "amount": 250000.0,
                    "currency": "INR",
                    "timestamp": "2024-05-31T18:00:00Z",
                    "status": "completed",
                    "channel": "online",
                    "description": "NEFT/5EDI/Salary Credit for AI Systems Engineering",
                    "risk_score": 0.02,
                    "source_system": "HDFC_CoreBanking",
                    "account_id": "ACC-HDFC-ASHIK-01",
                    "counterparty_org_id": "org-5edi-tech-001",
                    "fastrp_embedding": self.get_fastrp_dummy(),
                },
                {
                    "id": "TXN-ASHIK-IN-003",
                    "transaction_id": "TXN90481029386",
                    "type": "trade",
                    "amount": 450000.0,
                    "currency": "INR",
                    "timestamp": "2024-06-05T11:15:00Z",
                    "status": "completed",
                    "channel": "online",
                    "description": "NSE Trade Execution - Nifty 50 Index & IT ETF Long Allocation via Zerodha Kite",
                    "risk_score": 0.05,
                    "source_system": "Zerodha_Kite_API",
                    "account_id": "ACC-ZERODHA-ASHIK-02",
                    "counterparty_org_id": "org-zerodha-001",
                    "fastrp_embedding": self.get_fastrp_dummy(),
                },
                {
                    "id": "TXN-ASHIK-IN-004",
                    "transaction_id": "TXN90481029387",
                    "type": "transfer",
                    "amount": 95000.0,
                    "currency": "INR",
                    "timestamp": "2024-11-20T15:45:00Z",
                    "status": "flagged",
                    "channel": "mobile",
                    "description": "UPI/4120938491/Razorpay AI GPU Cloud Server Hosting Bengaluru",
                    "risk_score": 0.58,
                    "source_system": "HDFC_CoreBanking",
                    "account_id": "ACC-HDFC-ASHIK-01",
                    "counterparty_org_id": "org-razorpay-001",
                    "fastrp_embedding": self.get_fastrp_dummy(),
                },
                {
                    "id": "TXN-ASHIK-IN-005",
                    "transaction_id": "TXN90481029388",
                    "type": "transfer",
                    "amount": 120000.0,
                    "currency": "INR",
                    "timestamp": "2024-11-20T16:05:00Z",
                    "status": "completed",
                    "channel": "mobile",
                    "description": "UPI/4120938492/Koramangala Tech Park Co-Working Dedicated Space Lease",
                    "risk_score": 0.22,
                    "source_system": "HDFC_CoreBanking",
                    "account_id": "ACC-HDFC-ASHIK-01",
                    "counterparty_org_id": None,
                    "fastrp_embedding": self.get_fastrp_dummy(),
                },
                {
                    "id": "TXN-ASHIK-IN-006",
                    "transaction_id": "TXN90481029389",
                    "type": "transfer",
                    "amount": 300000.0,
                    "currency": "INR",
                    "timestamp": "2024-09-02T10:00:00Z",
                    "status": "completed",
                    "channel": "online",
                    "description": "IMPS/Transfer from HDFC to Zerodha Trading Margin Collateral Pool",
                    "risk_score": 0.04,
                    "source_system": "HDFC_CoreBanking",
                    "account_id": "ACC-HDFC-ASHIK-01",
                    "counterparty_org_id": "org-zerodha-001",
                    "fastrp_embedding": self.get_fastrp_dummy(),
                },
            ]

            for txn in transactions:
                session.run("""
                    MERGE (t:Transaction {id: $id})
                    SET t.transaction_id = $transaction_id,
                        t.type = $type,
                        t.amount = $amount,
                        t.currency = $currency,
                        t.timestamp = datetime($timestamp),
                        t.status = $status,
                        t.channel = $channel,
                        t.description = $description,
                        t.risk_score = $risk_score,
                        t.source_system = $source_system,
                        t.fastrp_embedding = $fastrp_embedding,
                        t.created_at = datetime()
                    WITH t
                    MATCH (a:Account {id: $account_id})
                    MERGE (a)-[:FROM_ACCOUNT]->(t)
                    WITH t
                    OPTIONAL MATCH (o:Organization {id: $counterparty_org_id})
                    FOREACH (_ IN CASE WHEN o IS NOT NULL THEN [1] ELSE [] END |
                        MERGE (t)-[:COUNTERPARTY]->(o)
                    )
                """, txn)

            # 7. Create Context Graph Decisions (Decision Traces)
            print("7. Creating Decision Traces with Semantic Reasoning & Gemini Embeddings...")
            decisions = [
                {
                    "id": "DEC-ASHIK-2024-001",
                    "decision_type": "approval",
                    "category": "compliance",
                    "status": "approved",
                    "timestamp": "2024-04-18T10:30:00Z",
                    "summary": "FEMA Inward Software Export Forex Remittance Approval ($12,500 USD from Vertex Global AI Labs).",
                    "reasoning": (
                        "Customer Ashik Hegde received an inward international SWIFT wire transfer of $12,500 USD (~₹10,40,000 INR) "
                        "from overseas counterparty Vertex Global AI Labs Inc (San Francisco, USA). Verified regulatory Purpose Code P0802 "
                        "(Software Consultancy & AI Engineering Services) supported by formal bilateral Master Services Agreement (MSA) "
                        "and tax invoice #5EDI-2024-04. OFAC sanctions screening and AML watchlist check returned 100% clear. "
                        "Electronic Form A2 & SOFTEX declaration documentation filed successfully. Compliance requirements under RBI FEMA Master Direction "
                        "fully satisfied. Cleared for immediate credit into ICICI Forex account with high confidence score of 0.98."
                    ),
                    "confidence": 0.98,
                    "risk_factors": ["foreign_currency_inward", "high_amount_cross_border"],
                    "source": "ICICI_TradeForex",
                    "about_account": "ACC-ICICI-ASHIK-03",
                    "made_by": "emp-priya-sharma-001",
                    "applied_policy": "pol-rbi-fema-001",
                    "fastrp_embedding": self.get_fastrp_dummy(),
                },
                {
                    "id": "DEC-ASHIK-2024-002",
                    "decision_type": "approval",
                    "category": "credit",
                    "status": "approved",
                    "timestamp": "2024-06-12T14:15:00Z",
                    "summary": "Prime Commercial Credit Limit & Visa Infinite Super-Premium Card Approval (₹15,00,000 limit).",
                    "reasoning": (
                        "Comprehensive credit underwriting evaluation conducted for customer Ashik Hegde. Credit bureau report reveals "
                        "an exemplary CIBIL score of 795 with zero 30+ DPD delinquencies across all historical credit instruments. "
                        "Bank statement reconciliation validates consistent monthly salary credits from 5edi Technologies Private Limited "
                        "and substantial foreign export consulting inflows (verified via Decision DEC-ASHIK-2024-001). "
                        "Debt-to-Income (DTI) ratio is exceptionally strong at 14.2%, well within the 35% policy ceiling. "
                        "Approved for Tier-1 Super-Premium unsecured credit limit of ₹15,00,000 under Tier-1 Commercial Credit Underwriting Policy."
                    ),
                    "confidence": 0.95,
                    "risk_factors": ["high_unsecured_credit_limit"],
                    "source": "HDFC_CreditRisk",
                    "about_account": "ACC-HDFC-ASHIK-01",
                    "made_by": "emp-vikram-malhotra-001",
                    "applied_policy": "pol-credit-tier1-001",
                    "fastrp_embedding": self.get_fastrp_dummy(),
                },
                {
                    "id": "DEC-ASHIK-2024-003",
                    "decision_type": "exception",
                    "category": "trading",
                    "status": "approved",
                    "timestamp": "2024-09-05T09:45:00Z",
                    "summary": "SEBI Margin Trading Facility Collateral Leverage Exception Approved (₹35,00,000 intraday cap).",
                    "reasoning": (
                        "Special trading risk committee evaluation for customer Ashik Hegde regarding Zerodha Demat & Trading Margin Account. "
                        "Customer requested an intraday leverage exposure exception increase from ₹15,00,000 to ₹35,00,000 for quantitative "
                        "algorithmic index strategies. Quantitative risk audit confirms liquid bluechip equity holdings (Nifty 50 constituents) "
                        "pledged as collateral with total haircut-adjusted margin coverage of 210% (surpassing the SEBI 150% threshold). "
                        "Account has maintained zero margin call breaches over 24 months of trading. Automated stop-loss algorithms confirmed active. "
                        "Exception granted under SEBI MTF Risk Policy."
                    ),
                    "confidence": 0.92,
                    "risk_factors": ["leverage_expansion", "intraday_derivative_exposure"],
                    "source": "Zerodha_RiskDesk",
                    "about_account": "ACC-ZERODHA-ASHIK-02",
                    "made_by": "emp-rohit-verma-001",
                    "applied_policy": "pol-sebi-mtf-001",
                    "fastrp_embedding": self.get_fastrp_dummy(),
                },
                {
                    "id": "DEC-ASHIK-2024-004",
                    "decision_type": "review",
                    "category": "fraud",
                    "status": "completed",
                    "timestamp": "2024-11-20T16:20:00Z",
                    "summary": "High-Velocity UPI Payout Alert Cleared as Legitimate B2B Cloud Infrastructure Expenses.",
                    "reasoning": (
                        "Automated fraud velocity alert ALT-IN-UPI-8821 triggered due to rapid consecutive outbound UPI transfers totaling "
                        "₹3,80,000 within 25 minutes from account ACC-HDFC-ASHIK-01. Deep graph and IP geolocation forensics confirm device "
                        "hardware fingerprint matches registered primary mobile handset in Koramangala, Bengaluru, Karnataka. "
                        "Counterparties identified as verified Indian tech infrastructure vendors (Razorpay software, dedicated server hosting). "
                        "Customer authenticated transactions via biometric 2FA. Alert classified as false positive; normal business operation verified."
                    ),
                    "confidence": 0.96,
                    "risk_factors": ["velocity_trigger", "rapid_successive_transfers"],
                    "source": "HDFC_FraudDesk",
                    "about_account": "ACC-HDFC-ASHIK-01",
                    "made_by": "emp-vikram-malhotra-001",
                    "applied_policy": "pol-upi-velocity-001",
                    "fastrp_embedding": self.get_fastrp_dummy(),
                },
            ]

            for dec in decisions:
                dec_emb = self.get_embedding(dec["summary"] + " " + dec["reasoning"])
                session.run("""
                    MERGE (d:Decision {id: $id})
                    SET d.decision_type = $decision_type,
                        d.category = $category,
                        d.status = $status,
                        d.decision_timestamp = datetime($timestamp),
                        d.reasoning = $reasoning,
                        d.reasoning_summary = $summary,
                        d.confidence_score = $confidence,
                        d.risk_factors = $risk_factors,
                        d.source_system = $source,
                        d.reasoning_embedding = $embedding,
                        d.fastrp_embedding = $fastrp_embedding,
                        d.created_at = datetime()
                    WITH d
                    MATCH (p:Person {id: 'person-ashik-hegde-001'})
                    MERGE (d)-[:ABOUT]->(p)
                    WITH d
                    MATCH (a:Account {id: $about_account})
                    MERGE (d)-[:ABOUT]->(a)
                    WITH d
                    MATCH (e:Employee {id: $made_by})
                    MERGE (d)-[:MADE_BY]->(e)
                    WITH d
                    MATCH (pol:Policy {id: $applied_policy})
                    MERGE (d)-[:APPLIED_POLICY]->(pol)
                """, {**dec, "embedding": dec_emb})

            # 8. Create Alerts & Support Tickets
            print("8. Creating Alerts & Support Tickets...")
            session.run("""
                MERGE (alt:Alert {id: 'ALT-IN-UPI-8821'})
                SET alt.alert_number = 'ALT-8821',
                    alt.alert_type = 'velocity_check',
                    alt.severity = 'medium',
                    alt.status = 'resolved',
                    alt.description = 'High-velocity outbound UPI transaction burst from Bangalore device',
                    alt.triggered_at = datetime('2024-11-20T15:50:00Z'),
                    alt.resolved_at = datetime('2024-11-20T16:20:00Z'),
                    alt.risk_score = 0.58,
                    alt.source_system = 'HDFC_FraudEngine',
                    alt.created_at = datetime()
                WITH alt
                MATCH (acc:Account {id: 'ACC-HDFC-ASHIK-01'})
                MERGE (alt)-[:REGARDING]->(acc)
                WITH alt
                MATCH (t:Transaction {id: 'TXN-ASHIK-IN-004'})
                MERGE (alt)-[:TRIGGERED_BY]->(t)
                WITH alt
                MATCH (d:Decision {id: 'DEC-ASHIK-2024-004'})
                MERGE (d)-[:RESOLVED_BY]->(alt)
                WITH alt
                MATCH (e:Employee {id: 'emp-vikram-malhotra-001'})
                MERGE (alt)-[:ASSIGNED_TO]->(e)
            """)

            session.run("""
                MERGE (s:SupportTicket {id: 'TKT-IN-5912'})
                SET s.ticket_number = 'TKT-5912',
                    s.subject = 'Request for Demat Margin Leverage Expansion for Algorithmic Trading Desk',
                    s.priority = 'medium',
                    s.status = 'closed',
                    s.description = 'Requesting increase in intraday margin trading facility limit backed by pledged Nifty bluechip collateral.',
                    s.submitted_at = datetime('2024-09-03T11:00:00Z'),
                    s.resolved_at = datetime('2024-09-05T09:45:00Z'),
                    s.created_at = datetime()
                WITH s
                MATCH (p:Person {id: 'person-ashik-hegde-001'})
                MERGE (s)-[:SUBMITTED_BY]->(p)
                WITH s
                MATCH (acc:Account {id: 'ACC-ZERODHA-ASHIK-02'})
                MERGE (s)-[:REGARDING]->(acc)
                WITH s
                MATCH (d:Decision {id: 'DEC-ASHIK-2024-003'})
                MERGE (d)-[:RESOLVED_BY]->(s)
                WITH s
                MATCH (e:Employee {id: 'emp-rohit-verma-001'})
                MERGE (s)-[:ASSIGNED_TO]->(e)
            """)

            # 9. Create Causal Chains Between Decisions
            print("9. Linking Causal Chains (CAUSED, INFLUENCED, PRECEDENT_FOR) between Decisions...")
            session.run("""
                MATCH (d1:Decision {id: 'DEC-ASHIK-2024-001'})
                MATCH (d2:Decision {id: 'DEC-ASHIK-2024-002'})
                MERGE (d1)-[r:INFLUENCED]->(d2)
                SET r.weight = 0.88,
                    r.influence_type = 'context',
                    r.notes = 'Verified foreign software export remittance cashflows served as financial stability proof for prime credit limit approval'
            """)

            session.run("""
                MATCH (d2:Decision {id: 'DEC-ASHIK-2024-002'})
                MATCH (d3:Decision {id: 'DEC-ASHIK-2024-003'})
                MERGE (d2)-[r:PRECEDENT_FOR]->(d3)
                SET r.similarity_score = 0.91,
                    r.outcome_relevance = 0.89,
                    r.notes = 'Prime credit standing and Tier-1 underwriting clearance provided precedent for approving expanded trading margin collateral leverage'
            """)

            session.run("""
                MATCH (d1:Decision {id: 'DEC-ASHIK-2024-001'})
                MATCH (d3:Decision {id: 'DEC-ASHIK-2024-003'})
                MERGE (d1)-[r:CAUSED]->(d3)
                SET r.confidence = 0.85,
                    r.causation_type = 'enabling',
                    r.notes = 'Liquidity from software consulting export remittances enabled capital deployment into Zerodha trading margin portfolio'
            """)

        print("\n" + "=" * 60)
        print("ASHIK HEGDE SEMANTIC DATA SEEDING COMPLETE!")
        print("=" * 60)
        print("Summary of data seeded:")
        print("  - Person: Ashik Hegde (Bengaluru, Karnataka, India)")
        print("  - Organizations: 5edi Technologies Pvt Ltd, HDFC Bank, Zerodha, ICICI, Razorpay, Vertex Global AI")
        print("  - Accounts: HDFC Checking, Zerodha Trading Demat, ICICI Forex EEFC Account")
        print("  - Policies: RBI FEMA Software Export, Tier-1 Prime Credit, SEBI MTF Leverage, NPCI UPI Velocity")
        print("  - Decisions: 4 Multi-Hop Decision Traces with full reasoning & 768-dim Gemini embeddings")
        print("  - Causal Chains: INFLUENCED, PRECEDENT_FOR, CAUSED relationships linking decisions")
        print("  - Transactions: SWIFT Forex wire, Salary credit, Demat stock purchase, UPI tech vendor payments")
        print("  - Alerts & Tickets: Resolved velocity alert and closed margin expansion ticket")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    seeder = AshikDataSeeder()
    try:
        seeder.seed_all()
    finally:
        seeder.close()

