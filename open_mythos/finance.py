"""
Finance Domain Adapters for OpenMythos.

Pre-built LoRA configurations for finance-specific tasks:
- Trading analysis (XAUUSD, forex, crypto)
- Business plan generation
- Ad copy optimization (Meta, Google, TikTok)
- Cashflow management
- Indonesian market specialization

Usage:
    from open_mythos.finance import FinanceAdapter, get_finance_adapter

    adapter = get_finance_adapter("trading")
    model = adapter.apply(model)
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
import logging

from open_mythos.lora import (
    LoRAConfig,
    apply_lora,
    save_lora_adapter,
    load_lora_adapter,
    merge_lora_weights,
)

logger = logging.getLogger(__name__)


@dataclass
class FinanceAdapterConfig:
    """
    Configuration for a finance domain adapter.

    Args:
        name: Adapter name (e.g., "trading", "business")
        description: Human-readable description
        lora_config: LoRA configuration
        training_data_path: Path to training data
        target_modules: Which layers to adapt
        special_tokens: Domain-specific tokens to add
    """

    name: str
    description: str
    lora_config: LoRAConfig
    training_data_path: Optional[str] = None
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    special_tokens: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Pre-built Finance Adapters
# ---------------------------------------------------------------------------

TRADING_ADAPTER = FinanceAdapterConfig(
    name="trading",
    description="Trading analysis: XAUUSD, forex, crypto, technical analysis, entry/exit signals",
    lora_config=LoRAConfig(
        rank=32,
        alpha=64,
        dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj"],
    ),
    special_tokens=[
        "[LONG]", "[SHORT]", "[ENTRY]", "[EXIT]", "[SL]", "[TP]",
        "[BULLISH]", "[BEARISH]", "[RSI]", "[MACD]", "[SMA]", "[EMA]",
        "[XAUUSD]", "[EURUSD]", "[GBPUSD]", "[USDJPY]", "[BTCUSD]",
    ],
)

BUSINESS_ADAPTER = FinanceAdapterConfig(
    name="business",
    description="Business plan generation, revenue models, market analysis, startup strategy",
    lora_config=LoRAConfig(
        rank=16,
        alpha=32,
        dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    ),
    special_tokens=[
        "[REVENUE]", "[COST]", "[PROFIT]", "[MARGIN]", "[CAC]", "[LTV]",
        "[TAM]", "[SAM]", "[SOM]", "[BURN_RATE]", "[RUNWAY]",
        "[PIVOT]", "[SCALE]", "[PMF]", "[GROWTH]",
    ],
)

ADS_ADAPTER = FinanceAdapterConfig(
    name="ads",
    description="Ad copy optimization for Meta, Google, TikTok ads",
    lora_config=LoRAConfig(
        rank=16,
        alpha=32,
        dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    ),
    special_tokens=[
        "[CTR]", "[CPC]", "[CPM]", "[ROAS]", "[CONVERSION]",
        "[HOOK]", "[CTA]", "[CREATIVE]", "[AUDIENCE]",
        "[META]", "[GOOGLE]", "[TIKTOK]", "[SHOPEE]",
    ],
)

CASHFLOW_ADAPTER = FinanceAdapterConfig(
    name="cashflow",
    description="Cashflow management, budgeting, financial planning",
    lora_config=LoRAConfig(
        rank=16,
        alpha=32,
        dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    ),
    special_tokens=[
        "[INFLOW]", "[OUTFLOW]", "[BALANCE]", "[BUDGET]",
        "[EMERGENCY_FUND]", "[INVESTMENT]", "[SAVINGS]",
        "[IDR]", "[USD]", "[EUR]",
    ],
)

INDONESIAN_MARKET_ADAPTER = FinanceAdapterConfig(
    name="indonesian_market",
    description="Indonesian market specialization: IDX, Shopee, Tokopedia, local regulations",
    lora_config=LoRAConfig(
        rank=16,
        alpha=32,
        dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    ),
    special_tokens=[
        "[IDX]", "[IHSG]", "[BBRI]", "[BBCA]", "[BMRI]",
        "[SHOPEE]", "[TOKOPEDIA]", "[LAZADA]", "[BLIBLI]",
        "[OJK]", "[BI_RATE]", "[INFLASI]", "[RUPIAH]",
    ],
)

# Registry of all adapters
FINANCE_ADAPTERS: Dict[str, FinanceAdapterConfig] = {
    "trading": TRADING_ADAPTER,
    "business": BUSINESS_ADAPTER,
    "ads": ADS_ADAPTER,
    "cashflow": CASHFLOW_ADAPTER,
    "indonesian_market": INDONESIAN_MARKET_ADAPTER,
}


class FinanceAdapter:
    """
    Finance domain adapter for OpenMythos.

    Wraps LoRA adapters with finance-specific configuration,
    training data, and special tokens.

    Args:
        config: Finance adapter configuration
    """

    def __init__(self, config: FinanceAdapterConfig):
        self.config = config
        self.model = None

    def apply(self, model: nn.Module) -> nn.Module:
        """Apply the finance adapter to a model."""
        model = apply_lora(model, self.config.lora_config)
        self.model = model
        logger.info(f"Applied '{self.config.name}' finance adapter")
        return model

    def save(self, path: str):
        """Save adapter weights."""
        if self.model is None:
            raise RuntimeError("No model to save. Call apply() first.")
        save_lora_adapter(self.model, path, config=self.config.lora_config)
        logger.info(f"Saved '{self.config.name}' adapter to {path}")

    def load(self, model: nn.Module, path: str) -> nn.Module:
        """Load adapter weights into a model."""
        model = self.apply(model)
        load_lora_adapter(model, path)
        self.model = model
        return model

    def get_training_prompt(self) -> str:
        """Get a training prompt template for this adapter."""
        templates = {
            "trading": (
                "Analyze {pair} for potential trading opportunity. "
                "Current price: {price}. Timeframe: {timeframe}. "
                "Provide entry, stop loss, and take profit levels."
            ),
            "business": (
                "Create a business plan for {business_type}. "
                "Target market: {market}. Initial investment: {budget}. "
                "Include revenue model, cost structure, and growth strategy."
            ),
            "ads": (
                "Write ad copy for {platform} campaign. "
                "Product: {product}. Target audience: {audience}. "
                "Budget: {budget}. Goal: {goal}."
            ),
            "cashflow": (
                "Analyze cashflow situation. "
                "Monthly revenue: {revenue}. Monthly expenses: {expenses}. "
                "Current balance: {balance}. Provide recommendations."
            ),
            "indonesian_market": (
                "Analyze {market} opportunity in Indonesia. "
                "Sector: {sector}. Target: {target}. "
                "Consider local regulations and market conditions."
            ),
        }
        return templates.get(self.config.name, "Analyze: {input}")

    def __repr__(self) -> str:
        return f"FinanceAdapter(name='{self.config.name}', description='{self.config.description}')"


def get_finance_adapter(name: str) -> FinanceAdapter:
    """
    Get a pre-built finance adapter by name.

    Args:
        name: Adapter name ("trading", "business", "ads", "cashflow", "indonesian_market")

    Returns:
        FinanceAdapter instance

    Raises:
        KeyError: If adapter not found
    """
    if name not in FINANCE_ADAPTERS:
        available = list(FINANCE_ADAPTERS.keys())
        raise KeyError(f"Adapter '{name}' not found. Available: {available}")

    return FinanceAdapter(FINANCE_ADAPTERS[name])


def list_finance_adapters() -> List[Dict[str, str]]:
    """List all available finance adapters."""
    return [
        {
            "name": config.name,
            "description": config.description,
            "rank": config.lora_config.rank,
            "alpha": config.lora_config.alpha,
        }
        for config in FINANCE_ADAPTERS.values()
    ]


def create_custom_adapter(
    name: str,
    description: str,
    rank: int = 16,
    alpha: int = 32,
    target_modules: Optional[List[str]] = None,
    special_tokens: Optional[List[str]] = None,
) -> FinanceAdapter:
    """
    Create a custom finance adapter.

    Args:
        name: Adapter name
        description: Human-readable description
        rank: LoRA rank
        alpha: LoRA alpha
        target_modules: Which layers to adapt
        special_tokens: Domain-specific tokens

    Returns:
        FinanceAdapter instance
    """
    config = FinanceAdapterConfig(
        name=name,
        description=description,
        lora_config=LoRAConfig(
            rank=rank,
            alpha=alpha,
            dropout=0.05,
            target_modules=target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"],
        ),
        special_tokens=special_tokens,
    )
    return FinanceAdapter(config)


# ---------------------------------------------------------------------------
# Training Data Generators
# ---------------------------------------------------------------------------

def generate_trading_data(output_path: str, num_samples: int = 100):
    """Generate trading analysis training data."""
    pairs = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "BTCUSD", "ETHUSD"]
    timeframes = ["1H", "4H", "Daily", "Weekly"]

    samples = []
    for i in range(num_samples):
        pair = pairs[i % len(pairs)]
        tf = timeframes[i % len(timeframes)]
        price = 2000 + (i * 10) % 500

        sample = {
            "text": (
                f"Trading Analysis [{pair}]\n"
                f"Timeframe: {tf}\n"
                f"Current Price: ${price}\n\n"
                f"Technical Indicators:\n"
                f"- RSI(14): {50 + (i % 30)} (neutral)\n"
                f"- MACD: {'bullish' if i % 2 == 0 else 'bearish'} crossover\n"
                f"- SMA(50): ${price - 20}\n"
                f"- SMA(200): ${price - 50}\n\n"
                f"Recommendation: {'LONG' if i % 3 == 0 else 'SHORT' if i % 3 == 1 else 'WAIT'}\n"
                f"Entry: ${price}\n"
                f"Stop Loss: ${price - 30}\n"
                f"Take Profit: ${price + 60}\n"
                f"Risk/Reward: 1:2"
            )
        }
        samples.append(sample)

    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    logger.info(f"Generated {num_samples} trading samples to {output_path}")


def generate_business_data(output_path: str, num_samples: int = 50):
    """Generate business plan training data."""
    businesses = [
        ("E-Commerce Platform", "IDR 500M", "Indonesian SMEs"),
        ("SaaS Product", "IDR 200M", "Startups"),
        ("Talent Agency", "IDR 100M", "Content creators"),
        ("Trading Fund", "IDR 1B", "Retail traders"),
        ("Digital Marketing Agency", "IDR 150M", "Local businesses"),
    ]

    samples = []
    for i in range(num_samples):
        biz = businesses[i % len(businesses)]
        sample = {
            "text": (
                f"Business Plan: {biz[0]}\n"
                f"Target Market: {biz[2]}\n"
                f"Initial Investment: {biz[1]}\n\n"
                f"Revenue Model:\n"
                f"- Primary: Subscription (IDR 100K/month)\n"
                f"- Secondary: Transaction fee (2.5%)\n"
                f"- Tertiary: Premium features (IDR 500K one-time)\n\n"
                f"Financial Projections:\n"
                f"- Year 1: IDR 1.2B revenue, IDR 800M expenses\n"
                f"- Year 2: IDR 3.6B revenue, IDR 2.0B expenses\n"
                f"- Year 3: IDR 8.4B revenue, IDR 4.2B expenses\n\n"
                f"Key Metrics:\n"
                f"- CAC: IDR 50K\n"
                f"- LTV: IDR 1.2M\n"
                f"- LTV/CAC: 24x\n"
                f"- Gross Margin: 75%"
            )
        }
        samples.append(sample)

    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    logger.info(f"Generated {num_samples} business samples to {output_path}")
