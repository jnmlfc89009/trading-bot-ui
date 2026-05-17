# 📈 Pairs Trading Pro Explorer

An interactive algorithmic trading dashboard built with Python and Streamlit. This application analyzes the statistical correlation between stock pairs, calculates real-time Z-Scores, and identifies potential mean-reversion trading signals.

## 🚀 Features
- **Dynamic Stock Analysis**: Compare any two tickers using live data from Yahoo Finance.
- **Z-Score Calculation**: Identify statistical outliers (overvalued/undervalued) in stock spreads.
- **Smart Logic**: Automatically detects price hierarchy and calculates optimal hedge ratios.
- **Professional Visualization**: Dual-subplot view featuring normalized price paths and Z-Score divergence.
- **Automated Signals**: Clear Buy/Sell instructions based on standard deviation thresholds.

## 🛠️ Tech Stack
- **Python 3.14**
- **Streamlit** (UI Framework)
- **Pandas/NumPy** (Data Processing)
- **Matplotlib** (Visualization)
- **yfinance** (Market Data API)

## 📦 Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/trading-bot-ui.git
   cd trading-bot-ui
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   streamlit run pair_explorer.py
   ```

## ⚖️ Disclaimer
This project is for educational purposes only. Algorithmic trading involves significant risk. Always perform your own due diligence before deploying capital.
