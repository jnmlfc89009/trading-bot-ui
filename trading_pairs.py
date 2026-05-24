# =====================================================================
# TRADING PAIRS DATABASE
# Add your favorite stock pairs here to track them automatically.
# =====================================================================

# Syntax:
# "UNIQUE_ID": {
#     "ticker_a": "SYMBOL1",
#     "ticker_b": "SYMBOL2",
#     "name":     "Human Readable Name",
#     "window":   <int>  # Rolling Z-score window in trading days
# }
#
# Window guidelines:
#   120 days → slow pairs (utilities, banks) driven by macro/rate cycles
#    90 days → medium pairs (financials, payments) with quarterly dynamics
#    60 days → fast pairs (tech, chips) where regimes shift quickly

APPROVED_PAIRS = {
    "US_UTILITIES": {
        "ticker_a": "DUK",
        "ticker_b": "SO",
        "name":     "Duke Energy vs Southern Co",
        "window":   120    # Slow utility cycle — driven by rate/regulatory changes
    },
    "US_PAYMENTS": {
        "ticker_a": "V",
        "ticker_b": "MA",
        "name":     "Visa vs Mastercard",
        "window":   90     # Medium speed — earnings & macro driven
    },
    "SG_BANKS": {
        "ticker_a": "D05.SI",
        "ticker_b": "O39.SI",
        "name":     "DBS vs OCBC",
        "window":   120    # Slow — driven by MAS policy and interest rate cycles
    },
    "AI_CHIPS": {
        "ticker_a": "NVDA",
        "ticker_b": "AMD",
        "name":     "Nvidia vs AMD",
        "window":   60     # Fast — tech sentiment shifts quickly
    },
}
