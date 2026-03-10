# Stockholm — Strategy & Vision

Deep research synthesis from Mistral AI + Perplexity, mapped to a 12-month execution roadmap.
Last updated: March 2026.

---

## North Star

> **"A silent guardian that runs 24/7, speaks only when it matters, proposes when to act, and requires your attention for maybe 10 minutes a week."**

Stockholm succeeds when you can ignore it for days and trust nothing important was missed.
It fails when it makes you open the dashboard to check on it.

Every feature decision must pass this test: *does this reduce cognitive load or increase it?*

The system should:
- Run all scans, geo checks, backtests, and learning cycles **autonomously**.
- Surface only **state changes** — new risk, direction flip, geo event, high-confidence signal — never re-assertions of what you already know.
- Deliver one weekly brief that covers *everything that changed* in 5 minutes.
- Propose a trade; you confirm in 10 seconds; it executes.
- Kill itself if the edge vanishes.

---

## Competitive Position

### What Stockholm already does that no competitor does

| Capability | Stockholm | Energent.ai | Kavout | LevelFields | OpenBB |
|---|---|---|---|---|---|
| Self-hosted, full data control | ✅ | ❌ SaaS | ❌ SaaS | ❌ SaaS | ⚠️ partial |
| Geopolitical risk per ticker | ✅ | ❌ | ❌ | ❌ | ❌ |
| Hard monthly EUR budget limits | ✅ | ❌ | ❌ | ❌ | ❌ |
| Adaptive self-learning + kill switch | ✅ | ❌ | ❌ | ❌ | ❌ |
| Quant + LLM + geo in one pipeline | ✅ | ⚠️ partial | ❌ | ❌ | ⚠️ partial |
| Portfolio geo exposure heatmap | ✅ | ❌ | ❌ | ❌ | ❌ |

**Energent.ai** claims 94.4% analytics accuracy but is a closed SaaS with no self-hosting. Strong on multimodal data, weak on geo depth and cost control.

**Kavout** covers 11,000+ stocks with AI agents but relies on proprietary models; no geopolitical or insider integration.

**LevelFields** focuses on event-driven alerts (SEC filings, earnings) but doesn't synthesize geopolitical or quant data into a unified score.

**Reflexivity** targets institutional investors with explainable AI but requires licensed data feeds.

**OpenBB** is a team workspace for analytics — not an autonomous, self-learning system.

### What Stockholm is missing vs. the market
- **Alternative data** (satellite imagery, shipping, social media at scale)
- **Regulatory compliance modules** (MiFID II, SEC AI risk disclosures)
- **Multi-agent pluggable architecture** (crypto, commodities, ESG agents)
- **Tax optimization** (AI-driven tax-loss harvesting, FIFO cost-basis tracking)

---

## 12-Month Phased Roadmap

### Phase 1 — Correctness First (Months 1–2)
*Nothing should run on real money if the math is wrong.*

These block everything else. Backtest alpha numbers are meaningless if cost basis is wrong,
FX P&L is invisible, or alert fatigue has made you stop trusting the system.

| Item | Why it blocks real money |
|---|---|
| Corporate actions ledger (#43) | Pre-split cost basis is silently wrong; P&L math is unreliable |
| Multi-currency support (#16) | EUR/USD mixed portfolio has invisible FX P&L error |
| Vectorized backtest with real costs (#32) | Gross vs net-of-costs is the only credible metric for trusting signals |
| Smarter alert dedup (#19) | Alert fatigue is the first thing that makes you stop trusting the system |
| Market holiday skip (#49) | Wasted budget + misleading "no news" analyses on closed days |
| Two-factor auth (#33) | Real API keys and broker access are coming — secure it first |
| DB backup rotation (#17) | ✅ done |

**Phase 1 exit criterion:** You can look at a signal that reads "58% win rate net of costs over 90 verified samples" and believe the number.

---

### Phase 2 — Geo + Macro Intelligence Completion (Months 2–4)
*The subsystem that makes Stockholm unique. Make it the best geo-risk engine in any self-hosted tool.*

| Item | Value |
|---|---|
| Geo history page + staleness invalidation (#6, #7) | Without history, geo is a black box; staleness-aware re-queue is your edge |
| Macro tracker: yield curve, VIX term, credit spreads (#22) | Rate environment shapes every sector signal; currently a blind spot |
| FOMC/ECB calendar + Stage 3 injection (#9) | Biggest single-day risk events, currently invisible to the pipeline |
| Geo scenario stress testing (#39) | Auto-runs "Taiwan scenario → -4.2% portfolio impact" when geo fires ≥8 |
| Supply chain risk mapping (#44) | Catches "TSMC affected → NVDA risk elevates" second-order exposure |
| Cross-asset composite signals (#47) | "Flight-to-safety: 4/4 aligned" is one clear signal vs 4 separate alerts |

**Phase 2 exit criterion:** When a ≥8 geo event fires, within 15 minutes the system has re-analyzed your full watchlist, stress-tested your portfolio against the matching scenario, assessed supply-chain second-order exposure, and fired **one** consolidated alert:

> *"Taiwan escalation detected. Portfolio estimated impact: -4.2%. NVDA, ASML geo-risk elevated. Full re-analysis complete."*

---

### Phase 3 — Autonomous Signal Confidence (Months 4–6)
*The system must earn trust from its own track record, not from your trust in the AI.*

| Item | Value |
|---|---|
| Feature importance from meta-labeler (#48) | Shows if RSI is actually predictive or noise in your signal set |
| Portfolio anomaly detection (#46) | Detects systemic moves vs idiosyncratic; blind spot of per-ticker approach |
| Earnings calendar full pipeline (#23) | Suppress STRONG_BUY 48h before earnings; track post-earnings drift |
| Correlation-aware position sizing (#35) | Kelly sizing that accounts for AAPL/MSFT correlation creep |
| Short squeeze scorer (#10) | Bear Case currently misses one of the biggest risk factors |
| 13F institutional tracker (#25) | Berkshire adds → confidence modifier, not just a footnote |

**Phase 3 exit criterion:** The meta-labeler tells you which quant factors drive accuracy in your universe. Position sizes are automatically adjusted for correlation. Earnings-week signals are suppressed without you thinking about it.

---

### Phase 4 — The "Don't Look At It" Layer (Months 6–9)
*This is the core goal. Automation that lets you safely ignore the system.*

**The weekly AI letter (#13) is the single highest-leverage item in the entire TODO.**

It replaces daily dashboard checking. It should include:
- What changed in portfolio risk this week (quant delta, geo delta, macro shift).
- Which signals fired and their verification status vs last week.
- Top 3 discovered opportunities.
- Learning stats: current hit rate, any kill-switch risks.
- One clear recommended action ("Consider reducing NVDA — Geo-Risiko rose 4→8, earnings in 3 days, RSI overbought").

| Item | Value for "don't look daily" |
|---|---|
| Weekly AI letter (#13) | **Replaces daily checking entirely** |
| Portfolio Q&A natural language (#37) | When you do look, ask one question instead of clicking through dashboards |
| Watchlist tiers + adaptive frequency (#12) | Tier 1 every cycle, Tier 3 weekly; reduces noise and cost |
| Continuous NLP scoring — VADER (#38) | Free hourly sentiment background scan; flags urgent tickers without burning budget |
| Push + Telegram bot (#14, #31) | **Urgent-only** (≥8 geo, direction flip, drawdown breach); silent otherwise |
| Watchlist groups / tags (#27) | Organize 40+ tickers by strategy/risk bucket without cognitive overhead |
| Confidence decay visualization (#28) | Stale signal = stale decision; surface it visually |

**Design constraint for Phase 4:** Push notifications and Telegram pings should fire fewer than once per day on average. If the system is noisy, the phase has failed.

**Phase 4 exit criterion:** You get a Telegram ping at most 1–2 times per week for genuinely new information. Sunday evening email letter covers everything in 5 minutes. You never open the dashboard except when you want to.

---

### Phase 5 — Real Money + Platform (Months 9–12)
*Only after Phases 1–4. The system has a verified paper track record, real costs modeled, geo/macro integrated. Now you pull the trigger.*

**Broker integration (#21) is strictly gated by Phases 1–3 being complete.**

The exact flow you want:

1. System generates STRONG_BUY, passes meta-labeler confidence ≥ 0.65.
2. Pre-flight checks: earnings not within 5 days, geo-risk < 6, no portfolio anomaly active, position within correlation-adjusted Kelly limit.
3. Telegram/push: *"Proposal: Buy 40 shares ASML at market. Entry ~€720. Stop-loss €612. [Bull case summary + sources]. Confirm? /yes /no"*
4. You reply `/yes`. One real order placed.
5. System tracks position, updates paper-vs-real P&L comparison.

This is not auto-trading. It is **AI proposes, human confirms in 10 seconds** — the right model for real money.

| Item | Platform / extensibility value |
|---|---|
| Personal API keys (#42) | Build Obsidian, Raycast, n8n, Notion automations on top of Stockholm |
| CLI mode (#34) | Script Stockholm from cron jobs, shell scripts, CI |
| Ollama LLM fallback (#41) | Budget-resilient; never stops working on the 15th of the month |
| PDF/CSV export (#36) | Tax records, audit trail, year-end review |
| Broker integration (#21) | The payoff of all the work above |

---

## Sequencing Logic

```
Phase 1 (M1–2)   → Trust the math
Phase 2 (M2–4)   → Trust the geo/macro intelligence
Phase 3 (M4–6)   → Trust the signal edge
Phase 4 (M6–9)   → Stop watching it daily    ← CORE GOAL
Phase 5 (M9–12)  → Put real money in it
```

You never skip a phase. Running Phase 5 on Phase 1 foundations is how paper-trade systems blow up with real money.

---

## What to De-Prioritize

Given "don't look at it daily" as the north star, some features work against the goal:

**Pairs trading (#40)** — intellectually interesting but adds a whole new strategy class to monitor. Goes against the "less attention needed" goal. Park for Year 2.

**Dark pool scraping (#26)** — brittle scrapers, high maintenance, uncertain signal quality for large-cap universe. Use 13F institutional data instead.

**Reddit / Google Trends (#24)** — meaningful for retail-driven small-caps but noisy and volatile. Pushes you toward more frequent checking, not less. Experiment later; don't make it core.

**Economic moat scoring (#45)** — great concept but moat scores are fairly stable for large-cap names. Not urgent; fine to add in Year 2.

---

## Future Trends to Watch (2026+)

From the Mistral AI research:

**Multi-agent pluggable architecture** — The future of AI investing is specialized agents (geo agent, quant agent, risk agent) that collaborate. Stockholm's pipeline already does this implicitly; a formal plugin architecture would let you add crypto, commodities, or ESG agents without touching core code.

**Edge AI / local models** — As on-device models improve, more of Stockholm's inference can move fully local. Ollama fallback (#41) is the first step; a fully offline mode (no API costs at all) is a 2027 target.

**Tokenized assets** — Blockchain-based securities are rising. Worth watching but not in scope for 12 months.

**Synthetic data for black swan modeling** — Platforms like Magnifi use synthetic crisis scenarios to stress-test portfolios. Stockholm's geo scenario engine (#39) is a manual version of this; synthetic data generation is the next evolution.

**Regulatory pressure** — AI in finance faces tightening rules (MiFID II, SEC AI risk disclosures). Adding a basic compliance audit log (what AI recommended, what you acted on, why) is worth doing before real money is involved.

---

## Open Strategic Questions

1. **Monetization without compromising self-hosted ethos** — Premium plugins? Enterprise support contracts? Hosted version for non-technical users?
2. **Open-sourcing the pipeline** — Would community contributions (more data connectors, alternative LLM backends, backtesting strategies) outweigh the loss of competitive moat?
3. **A "lite" cloud version** — Attracts users who don't want to self-host; creates a funnel toward the full self-hosted version for power users.
4. **Academic publication** — By Phase 3, Stockholm will have a dataset rich enough (geo events → signal outcomes → verified predictions) to write something genuinely novel about AI-assisted geopolitical risk-aware stock analysis.

---

## One-Line Summary Per Phase

| Phase | What you build | What you get |
|---|---|---|
| 1 | Correctness foundation | Numbers you can trust |
| 2 | Geo + macro intelligence | Context for every signal |
| 3 | Self-auditing signal lab | Edge you can prove |
| 4 | Silent guardian UX | Freedom from daily monitoring |
| 5 | Real money + platform | Stockholm earns its place |
