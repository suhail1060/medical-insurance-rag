---
title: "Real-Time Claim Status Tracking and Adjudication Lifecycle"
category: claims
doc_id: claims_07
topics: [claim-status-tracking, claim-lifecycle, edi-276-277, member-dashboard]
---

Policyholders and healthcare providers can monitor the processing lifecycle of medical claims through real-time tracking systems provided by Apex Health Plans. This document outlines the primary stages of claim adjudication.

Claim Lifecycle Stages:
1. Received / Ingestion:
The claim enters the clearinghouse via EDI 837 electronic transaction or physical mail scanning. An internal Claim Reference Number (CRN)—a unique 12-digit tracking identifier—is assigned within twenty-four (24) hours of receipt.

2. In Adjudication / Automated Review:
The claim is processed through automated claims editing systems. The software verifies member eligibility, checks active coverage dates, validates CPT and ICD-10 code combinations, and calculates deductible and coinsurance accumulators.

3. Suspended / Information Requested (Pended):
If automated edits detect missing records, lack of prior authorization, or coordination of benefits questions, the claim is placed in 'Suspended' status. A formal notice is transmitted electronically (EDI 277) or mailed to the provider and member specifying the required documents.

4. Processed / Finalized:
Clinical and financial adjudication is complete. A final determination is rendered: Approved, Partially Approved, or Denied. The system generates an Explanation of Benefits (EOB) for the member and an Electronic Remittance Advice (ERA / EDI 835) for the provider.

5. Paid / Settled:
Payment is released. In-network providers receive funds via Electronic Funds Transfer (EFT). Members receiving direct reimbursement receive an automated direct deposit or a paper check mailed within three business days.

Methods for Tracking Claim Status:
- Online Member Portal: Log into www.apexhealthplans.com/claims to view claims updated every 24 hours.
- Automated Telephone System: Call 1-800-555-APEX and enter your 10-digit Member ID and date of service to hear real-time status.
- Provider EDI Status Inquiry: Provider billing offices can transmit EDI 276 real-time status transactions to receive instant EDI 277 status responses.
