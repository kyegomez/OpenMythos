"""
Generate comprehensive finance training dataset for OpenMythos.

Creates 500+ training samples covering:
- Technical analysis & trading signals
- Business plan generation
- Ad copy optimization
- Cashflow management
- Indonesian market specifics
- Risk management
- Portfolio optimization

Output: JSONL format for LoRA fine-tuning
"""

import json
import random
from pathlib import Path

random.seed(42)

# ---------------------------------------------------------------------------
# Trading Analysis Templates
# ---------------------------------------------------------------------------

TRADING_PAIRS = [
    ("XAUUSD", "Gold", 2350, 50),
    ("EURUSD", "Euro/Dollar", 1.0850, 0.0050),
    ("GBPUSD", "Pound/Dollar", 1.2700, 0.0080),
    ("USDJPY", "Dollar/Yen", 155.50, 1.50),
    ("BTCUSD", "Bitcoin", 67500, 5000),
    ("ETHUSD", "Ethereum", 3200, 400),
    ("USDIDR", "Dollar/Rupiah", 15850, 200),
    ("AUDUSD", "Aussie/Dollar", 0.6650, 0.0040),
]

TIMEFRAMES = ["15M", "1H", "4H", "Daily", "Weekly"]
INDICATORS = ["RSI", "MACD", "SMA", "EMA", "Bollinger", "Stochastic", "ADX", "Ichimoku"]
SIGNALS = ["LONG", "SHORT", "WAIT", "HOLD"]
RISK_LEVELS = ["Low", "Medium", "High"]


def gen_trading_analysis():
    """Generate trading analysis samples."""
    samples = []
    
    for pair, name, base_price, volatility in TRADING_PAIRS:
        for _ in range(15):
            tf = random.choice(TIMEFRAMES)
            price = base_price + random.uniform(-volatility, volatility)
            signal = random.choice(SIGNALS)
            rsi = random.randint(20, 80)
            
            if pair in ["XAUUSD", "BTCUSD", "ETHUSD"]:
                price_str = f"${price:,.2f}"
                sl = price - volatility * 0.3
                tp = price + volatility * 0.6
            elif pair == "USDJPY":
                price_str = f"{price:.2f}"
                sl = price - 0.50
                tp = price + 1.00
            elif pair == "USDIDR":
                price_str = f"IDR {price:,.0f}"
                sl = price - 100
                tp = price + 200
            else:
                price_str = f"{price:.4f}"
                sl = price - 0.0020
                tp = price + 0.0040
            
            macd_state = random.choice(["bullish crossover", "bearish crossover", "neutral", "divergence"])
            trend = random.choice(["uptrend", "downtrend", "sideways", "consolidation"])
            support = price - volatility * random.uniform(0.2, 0.5)
            resistance = price + volatility * random.uniform(0.2, 0.5)
            risk_pct = random.choice([0.5, 1.0, 1.5, 2.0])
            
            fmt = ',.0f' if pair == 'USDIDR' else ',.2f'
            
            text = f"""## Trading Analysis: {pair} ({name})

**Timeframe:** {tf}
**Current Price:** {price_str}
**Trend:** {trend}

### Technical Indicators
- RSI(14): {rsi} ({'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'})
- MACD: {macd_state}
- Support: {support:{fmt}}
- Resistance: {resistance:{fmt}}

### Signal
- Direction: **{signal}**
- Entry: {price_str}
- Stop Loss: {sl:{fmt}}
- Take Profit: {tp:{fmt}}
- Risk: {risk_pct}% of account
- Risk/Reward: 1:{tp/price*100:.1f}

### Analysis
{'Bullish momentum building. Price above key moving averages. RSI shows room for upside.' if signal == 'LONG' else 'Bearish pressure increasing. Price below key moving averages. RSI indicates weakness.' if signal == 'SHORT' else 'No clear directional bias. Wait for breakout or confirmation before entering.' if signal == 'WAIT' else 'Position in profit. Trail stop loss to breakeven. Consider partial profit taking.'}

### Risk Management
- Position size: {(risk_pct / 1.5):.1f}% of capital
- Maximum drawdown: 5% per trade
- Correlation check: {'Low correlation with existing positions' if random.random() > 0.5 else 'High correlation - reduce position size'}
"""
            samples.append({"text": text.strip()})
    
    return samples


# ---------------------------------------------------------------------------
# Business Plan Templates
# ---------------------------------------------------------------------------

BUSINESS_TYPES = [
    ("E-Commerce Platform", "IDR 500M", "Indonesian MSMEs", "SaaS + Transaction fees"),
    ("Talent Agency", "IDR 200M", "Content creators", "Commission-based"),
    ("Trading Fund", "IDR 1B", "Retail traders", "Performance fees"),
    ("Digital Marketing Agency", "IDR 150M", "Local businesses", "Retainer + Performance"),
    ("AI Software House", "IDR 300M", "Enterprise clients", "Project-based + SaaS"),
    ("Food & Beverage Chain", "IDR 800M", "Urban millennials", "Direct sales + Franchise"),
    ("EdTech Platform", "IDR 250M", "Students & professionals", "Subscription"),
    ("Logistics Startup", "IDR 600M", "E-commerce sellers", "Per-delivery fee"),
]


def gen_business_plans():
    """Generate business plan samples."""
    samples = []
    
    for biz_name, investment, market, revenue_model in BUSINESS_TYPES:
        for i in range(5):
            year1_rev = random.randint(500, 3000)
            year2_rev = year1_rev * random.uniform(2.0, 4.0)
            year3_rev = year2_rev * random.uniform(1.5, 3.0)
            margin = random.randint(60, 85)
            cac = random.randint(30, 150)
            ltv = cac * random.uniform(3, 15)
            
            text = f"""## Business Plan: {biz_name}

### Executive Summary
{biz_name} targets {market} with a {revenue_model} model. Initial investment: {investment}.

### Market Opportunity
- TAM: IDR {random.randint(10, 500)}T
- SAM: IDR {random.randint(1, 50)}T
- SOM: IDR {random.randint(100, 5000)}B
- Growth rate: {random.randint(15, 45)}% CAGR

### Revenue Model
**Primary:** {revenue_model}
- Year 1: IDR {year1_rev}M
- Year 2: IDR {year2_rev:,.0f}M
- Year 3: IDR {year3_rev:,.0f}M

### Cost Structure
- Fixed costs: IDR {random.randint(50, 200)}M/month
- Variable costs: {random.randint(20, 40)}% of revenue
- Gross margin: {margin}%

### Unit Economics
- CAC: IDR {cac}K
- LTV: IDR {ltv:,.0f}K
- LTV/CAC: {ltv/cac:.1f}x
- Payback period: {random.randint(3, 12)} months

### Growth Strategy
1. Phase 1 (0-6 months): Product-market fit, first 100 customers
2. Phase 2 (6-18 months): Scale marketing, expand to 3 cities
3. Phase 3 (18-36 months): Regional expansion, raise Series A

### Key Risks
- Competition from established players
- Regulatory changes in {market.split()[0]} sector
- Cash runway: {random.randint(6, 18)} months at current burn rate

### Team
- CEO: Domain expert with {random.randint(5, 15)} years experience
- CTO: Full-stack engineer, previous startup exit
- Head of Growth: Digital marketing specialist
"""
            samples.append({"text": text.strip()})
    
    return samples


# ---------------------------------------------------------------------------
# Ad Copy Templates
# ---------------------------------------------------------------------------

PLATFORMS = ["Meta Ads", "Google Ads", "TikTok Ads", "Shopee Ads"]
PRODUCTS = [
    "Trading Course", "Business Template Pack", "AI Tool", "Marketing Ebook",
    "SaaS Product", "Coaching Program", "Digital Product", "Membership",
]
AUDIENCES = [
    "Young entrepreneurs 25-35",
    "Retail traders",
    "Small business owners",
    "Digital marketers",
    "Freelancers",
    "Side hustlers",
]


def gen_ad_copy():
    """Generate ad copy optimization samples."""
    samples = []
    
    for platform in PLATFORMS:
        for _ in range(8):
            product = random.choice(PRODUCTS)
            audience = random.choice(AUDIENCES)
            ctr = random.uniform(0.8, 4.5)
            cpc = random.randint(500, 8000)
            roas = random.uniform(1.5, 8.0)
            
            hooks = [
                f"Stop losing money on trades. Here's the {product} that changed everything.",
                f"I made IDR 50M in 30 days with this {product}. Here's how.",
                f"97% of {audience.split()[0]}s don't know this {product} trick.",
                f"The {product} that top performers don't want you to know about.",
                f"How I went from zero to IDR 100M/month using this {product}.",
            ]
            
            ctas = [
                "Get Started Now →",
                "Download Free Guide",
                "Claim Your Spot",
                "Learn More",
                "Start Your Free Trial",
            ]
            
            text = f"""## Ad Copy: {platform} Campaign

**Product:** {product}
**Target Audience:** {audience}
**Budget:** IDR {random.randint(5, 50)}M/month

### Headline Options
1. {random.choice(hooks)}
2. {random.choice(hooks)}
3. {random.choice(hooks)}

### Primary Text
Are you tired of {random.choice(['struggling with', 'guessing about', 'losing money on'])} {product.lower()}s?

Our {product} has helped {random.randint(100, 10000)}+ {audience.lower()}s achieve {random.choice(['consistent profits', 'business growth', 'financial freedom'])}.

✅ {random.choice(['Proven system', 'Step-by-step guide', 'AI-powered'])}
✅ {random.choice(['No experience needed', 'Works in any market', 'Results in 7 days'])}
✅ {random.choice(['Money-back guarantee', 'Free trial available', 'Lifetime access'])}

### CTA
{random.choice(ctas)}

### Performance Metrics (Previous Campaign)
- CTR: {ctr:.2f}%
- CPC: IDR {cpc:,}
- ROAS: {roas:.1f}x
- Conversion Rate: {random.uniform(1, 8):.1f}%

### Optimization Tips
- {'Test 3-5 creative variations' if platform == 'Meta Ads' else 'Focus on search intent keywords' if platform == 'Google Ads' else 'Use trending sounds and quick hooks' if platform == 'TikTok Ads' else 'Optimize product images and titles'}
- Retarget engaged audiences for 2x conversion
- Scale winning ads by 20% every 3 days
"""
            samples.append({"text": text.strip()})
    
    return samples


# ---------------------------------------------------------------------------
# Cashflow & Financial Planning
# ---------------------------------------------------------------------------

def gen_cashflow():
    """Generate cashflow management samples."""
    samples = []
    
    scenarios = [
        ("Startup", 200, 180, 500),
        ("Growing Business", 800, 600, 2000),
        ("Freelancer", 30, 20, 100),
        ("Trading Account", 500, 50, 5000),
        ("Side Hustle", 15, 10, 50),
    ]
    
    for scenario, revenue, expenses, balance in scenarios:
        for _ in range(5):
            rev_var = revenue * random.uniform(0.8, 1.2)
            exp_var = expenses * random.uniform(0.9, 1.1)
            net = rev_var - exp_var
            months_runway = balance / max(abs(net), 1) if net < 0 else 999
            
            text = f"""## Cashflow Analysis: {scenario}

### Monthly Summary
- Revenue: IDR {rev_var:,.0f}M
- Expenses: IDR {exp_var:,.0f}M
- Net Cashflow: IDR {net:,.0f}M {'✅ Positive' if net > 0 else '⚠️ Negative'}
- Current Balance: IDR {balance:,.0f}M

### Expense Breakdown
- Operations: IDR {exp_var * 0.4:,.0f}M ({40}%)
- Marketing: IDR {exp_var * 0.25:,.0f}M ({25}%)
- Personnel: IDR {exp_var * 0.25:,.0f}M ({25}%)
- Other: IDR {exp_var * 0.1:,.0f}M ({10}%)

### Projections
- 3-month runway: {'Sufficient' if months_runway > 6 else '⚠️ Action needed'}
- Break-even: {random.randint(2, 12)} months
- Annual projection: IDR {net * 12:,.0f}M

### Recommendations
{'1. Increase marketing ROI - focus on organic channels\n2. Reduce operational costs by 10-15%\n3. Build emergency fund (3-6 months expenses)\n4. Diversify revenue streams' if net < 0 else '1. Maintain current trajectory\n2. Invest surplus in growth initiatives\n3. Build 6-month emergency fund\n4. Consider strategic investments'}

### Action Items
- [ ] Review and cut unnecessary subscriptions
- [ ] Negotiate better terms with suppliers
- [ ] Set up automated savings (20% of revenue)
- [ ] Track daily cashflow in spreadsheet
"""
            samples.append({"text": text.strip()})
    
    return samples


# ---------------------------------------------------------------------------
# Indonesian Market Specifics
# ---------------------------------------------------------------------------

def gen_indonesian_market():
    """Generate Indonesian market analysis samples."""
    samples = []
    
    sectors = [
        ("IDX", "IHSG", "Banking", ["BBRI", "BBCA", "BMRI"]),
        ("Crypto", "Bitcoin", "P2P Trading", ["Bitcoin", "Ethereum", "USDT"]),
        ("E-Commerce", "Shopee", "Marketplace", ["Shopee", "Tokopedia", "Lazada"]),
        ("Property", "Residential", "Investment", ["Jakarta", "Surabaya", "Bandung"]),
    ]
    
    for sector, market, focus, assets in sectors:
        for _ in range(5):
            text = f"""## Indonesian Market Analysis: {sector}

### Market Overview
- Sector: {sector} ({focus})
- Market: {market}
- Key Assets: {', '.join(assets)}

### Current Conditions
- IHSG: {random.randint(7000, 7500)} (YTD: {random.uniform(-5, 15):.1f}%)
- Rupiah: IDR {random.randint(15500, 16200)}/USD
- BI Rate: {random.uniform(5.5, 6.5):.2f}%
- Inflation: {random.uniform(2.5, 4.5):.1f}%

### Opportunities
1. {random.choice(['Banking sector outperforming', 'Crypto adoption growing', 'E-commerce penetration increasing'])}
2. {random.choice(['Government infrastructure spending', 'Digital economy expansion', 'Foreign investment inflows'])}
3. {random.choice(['Demographic dividend', 'Rising middle class', 'Tech startup ecosystem'])}

### Risks
1. {random.choice(['Global recession fears', 'USD strength', 'Geopolitical tensions'])}
2. {random.choice(['Regulatory uncertainty', 'Interest rate volatility', 'Commodity price swings'])}
3. {random.choice(['Domestic political risk', 'Natural disasters', 'Supply chain disruptions'])}

### Strategy
- Allocation: {random.randint(30, 60)}% domestic, {random.randint(40, 70)}% international
- Focus: {random.choice(['Value stocks with strong dividends', 'Growth stocks in tech sector', 'Balanced portfolio with bonds'])}
- Risk: Medium (diversified across {random.randint(3, 8)} positions)

### Regulatory Notes
- OJK regulates financial services
- Bappebti regulates crypto/commodity trading
- Tax: 0.1% on stock transactions, progressive on gains
"""
            samples.append({"text": text.strip()})
    
    return samples


# ---------------------------------------------------------------------------
# Risk Management & Portfolio
# ---------------------------------------------------------------------------

def gen_risk_management():
    """Generate risk management and portfolio samples."""
    samples = []
    
    for _ in range(15):
        portfolio_size = random.choice([10, 50, 100, 500, 1000])
        stocks_pct = random.randint(30, 60)
        crypto_pct = random.randint(10, 30)
        bonds_pct = random.randint(10, 25)
        cash_pct = 100 - stocks_pct - crypto_pct - bonds_pct
        
        sharpe = random.uniform(0.5, 2.5)
        max_dd = random.randint(5, 25)
        
        text = f"""## Portfolio Review & Risk Management

### Portfolio Summary
- Total Value: ${portfolio_size:,}K
- YTD Return: {random.uniform(-10, 30):.1f}%
- Sharpe Ratio: {sharpe:.2f}
- Max Drawdown: {max_dd}%

### Asset Allocation
- Stocks: {stocks_pct}% (${portfolio_size * stocks_pct / 100:,.0f}K)
- Crypto: {crypto_pct}% (${portfolio_size * crypto_pct / 100:,.0f}K)
- Bonds: {bonds_pct}% (${portfolio_size * bonds_pct / 100:,.0f}K)
- Cash: {cash_pct}% (${portfolio_size * cash_pct / 100:,.0f}K)

### Risk Metrics
- VaR (95%): ${portfolio_size * 0.02:,.0f}K
- Beta: {random.uniform(0.7, 1.3):.2f}
- Volatility: {random.uniform(10, 30):.1f}%
- Correlation to S&P500: {random.uniform(0.3, 0.9):.2f}

### Recommendations
{'1. Rebalance: Reduce crypto to 15%, increase bonds to 25%\n2. Add international exposure (20%)\n3. Set stop-loss at 15% for individual positions\n4. Review quarterly' if crypto_pct > 25 else '1. Current allocation is well-balanced\n2. Consider adding alternative assets (REITs, commodities)\n3. Maintain stop-loss discipline\n4. Tax-loss harvest before year end'}

### Stress Test Scenarios
- 2008-style crash: Portfolio would lose ~{max_dd * 1.5:.0f}%
- Rate hike +2%: Bonds lose ~{bonds_pct * 0.3:.0f}%, stocks flat
- Crypto winter: Crypto loses ~{crypto_pct * 0.8:.0f}%, rest stable
"""
        samples.append({"text": text.strip()})
    
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_dir = Path("data/finance")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_samples = []
    
    # Generate all types
    print("Generating trading analysis samples...")
    trading = gen_trading_analysis()
    all_samples.extend(trading)
    print(f"  → {len(trading)} samples")
    
    print("Generating business plan samples...")
    business = gen_business_plans()
    all_samples.extend(business)
    print(f"  → {len(business)} samples")
    
    print("Generating ad copy samples...")
    ads = gen_ad_copy()
    all_samples.extend(ads)
    print(f"  → {len(ads)} samples")
    
    print("Generating cashflow samples...")
    cashflow = gen_cashflow()
    all_samples.extend(cashflow)
    print(f"  → {len(cashflow)} samples")
    
    print("Generating Indonesian market samples...")
    indonesia = gen_indonesian_market()
    all_samples.extend(indonesia)
    print(f"  → {len(indonesia)} samples")
    
    print("Generating risk management samples...")
    risk = gen_risk_management()
    all_samples.extend(risk)
    print(f"  → {len(risk)} samples")
    
    # Shuffle
    random.shuffle(all_samples)
    
    # Save as JSONL
    output_path = output_dir / "finance_dataset.jsonl"
    with open(output_path, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")
    
    print(f"\n✅ Total: {len(all_samples)} training samples")
    print(f"📁 Saved to: {output_path}")
    
    # Also save splits
    split_idx = int(len(all_samples) * 0.9)
    train = all_samples[:split_idx]
    val = all_samples[split_idx:]
    
    with open(output_dir / "train.jsonl", "w") as f:
        for s in train:
            f.write(json.dumps(s) + "\n")
    
    with open(output_dir / "val.jsonl", "w") as f:
        for s in val:
            f.write(json.dumps(s) + "\n")
    
    print(f"📊 Train: {len(train)} | Val: {len(val)}")


if __name__ == "__main__":
    main()
