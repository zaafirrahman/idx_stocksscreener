# 📈 IDX Stocks Screener (Market Flow Strategy)

This strategy aims to identify high-probability market opportunities using a staged approach based on screening, flow analysis, and price confirmation before scaling positions.

The approach focuses on minimizing initial risk and increasing exposure only after market direction is validated.

---

## ⚡ Strategy Workflow

### 1️⃣ Freq Analyzer → Candidate Shortlist

Initial screening using a frequency analyzer to identify assets with significant activity and potential price movement.

**Purpose:**

* Filter active markets
* Reduce noise
* Focus on early momentum candidates

**Output:** shortlist watchlist.

---

### 2️⃣ Flow Indicator → Small Entry (Probe Position)

Uses flow indicators to read buying/selling pressure and initiate a small entry as an early position.

**Purpose:**

* Test market reaction
* Maintain minimal risk exposure
* Early positioning if momentum is valid

**Position size:** small (probe entry).

---

### 3️⃣ Price Confirmation → Alert Trigger

Waits for price movement to confirm the initial bias before increasing exposure.

**Purpose:**

* Validate market direction
* Avoid false signals
* Enter based on price behavior rather than prediction

**Result:** alert triggered when price confirms.

---

### 4️⃣ Add Position → Ride Markup

Scales into the position once the movement proves valid and momentum continues.

**Purpose:**

* Scale into strength
* Maximize momentum
* Ride the trend during the markup phase

---

## 🎯 Core Principles

* **Screening → Validation → Confirmation → Scaling**
* Small initial risk, increased exposure after validation
* Price action as primary confirmation
* Momentum-driven positioning

---

## ⚠️ Status

This project is currently under development and testing
