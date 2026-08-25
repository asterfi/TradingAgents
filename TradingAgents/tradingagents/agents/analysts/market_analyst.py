from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_indicators,
    get_instrument_context_from_state,
    get_language_instruction,
    get_stock_data,
    get_verified_market_snapshot,
)
from tradingagents.dataflows.shadow_snapshot import (
    build_grounding_block,
    build_stock_grounding_block,
)

SHADOW_STOCK_MARKET_PROMPT = (
    "You are the Market Analyst for a US equity watchlist (NVDA/TSLA/AAPL/AMD/SPY). "
    "Your ENTIRE input is the stock snapshot below. Analyze ONLY the figures present:\n"
    "- trend (close vs sma50/sma200) and position vs 20d/52w extremes\n"
    "- key levels: prior-day pivots (P/R1/R2/S1/S2), gap vs prior close, session VWAP\n"
    "- volatility (ATR14) and volume pace vs 20d average (confirmation)\n"
    "- earnings date (event risk)\n"
    "RULES (hard):\n"
    "1. Cite ONLY figures present in the snapshot. Do not compute, estimate, or invent any "
    "indicator, level, percentage, or number not listed (no RSI, no Bollinger, no MACD, no "
    "derived pivots beyond those listed).\n"
    "2. Ratios or distances computed from two snapshot figures (e.g. % above SMA, % to pivot) "
    "are allowed ONLY when explicitly labeled '(derived from snapshot)'.\n"
    "3. Do not fetch any data. You have no tools.\n"
    "4. If a figure is not in the snapshot, say 'not in snapshot' rather than guessing.\n"
    "5. State your read of the level geometry factually; do not make a directional prediction.\n"
    "Write a detailed report of the observable structure with a Markdown table of the key "
    "snapshot fields at the end."
)

SHADOW_MARKET_PROMPT = (
    "You are the Market Analyst for a deterministic breakout system (btc-swing). "
    "Your ENTIRE input is the btc-swing snapshot below. Analyze ONLY the figures present:\n"
    "- trend (close vs sma50), close vs high20/low20 breakout geometry and distance to triggers\n"
    "- vol_ratio and volume_24h_usd (volume confirmation)\n"
    "- funding rate (crowding), macro regime, scheduled events (blackout risk)\n"
    "- price_now vs close (current drift), open position context if any\n"
    "RULES (hard):\n"
    "1. Cite ONLY figures present in the snapshot. Do not compute, estimate, or invent any "
    "indicator, level, percentage, or number not listed (no RSI, no Bollinger, no MACD, no "
    "derived pivots).\n"
    "2. Ratios or distances computed from two snapshot figures (e.g. % above SMA, % to trigger) "
    "are allowed ONLY when explicitly labeled '(derived from snapshot)'.\n"
    "3. Do not fetch any data. You have no tools.\n"
    "4. If a figure is not in the snapshot, say 'not in snapshot' rather than guessing.\n"
    "5. State your read of the setup geometry factually; do not make a directional prediction.\n"
    "Write a detailed report of the observable structure with a Markdown table of the key "
    "snapshot fields at the end."
)


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)
        shadow = state.get("shadow_snapshot")

        if shadow is not None:
            # ---- SHADOW MODE: snapshot-only, no tools, no external data ----
            snapshot, symbol = shadow["snapshot"], shadow["symbol"]
            if shadow.get("kind") == "stock":
                block = build_stock_grounding_block(snapshot,
                                                    variant=shadow.get("variant", "A"))
                opctx = shadow.get("operator_context") or ""
                system_message = (
                    SHADOW_STOCK_MARKET_PROMPT + "\n\n" + block
                    + ("\n" + opctx if opctx else "")
                    + "\n" + get_language_instruction()
                )
            else:
                block = build_grounding_block(snapshot, symbol,
                                              variant=shadow.get("variant", "A"),
                                              position_blind=bool(shadow.get("position_blind", False)))
                system_message = (
                    SHADOW_MARKET_PROMPT + "\n\n" + block + "\n" + get_language_instruction()
                )
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", "You are a helpful AI assistant, collaborating with other assistants. "
                     "Today's date is {current_date}; treat it as 'now'. {instrument_context}\n{system_message}"),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            prompt = prompt.partial(system_message=system_message)
            prompt = prompt.partial(current_date=current_date)
            prompt = prompt.partial(instrument_context=instrument_context)
            chain = prompt | llm
            result = chain.invoke(state["messages"])
            report = result.content if isinstance(result.content, str) else str(result.content)
            return {"messages": [result],
                    "market_report": report}

        tools = [
            get_stock_data,
            get_indicators,
            get_verified_market_snapshot,
        ]

        system_message = (
            """You are a trading assistant tasked with analyzing financial markets. Your role is to select the **most relevant indicators** for a given market condition or trading strategy from the following list. The goal is to choose up to **8 indicators** that provide complementary insights without redundancy. Categories and each category's indicators are:

Moving Averages:
- close_50_sma: 50 SMA: A medium-term trend indicator. Usage: Identify trend direction and serve as dynamic support/resistance. Tips: It lags price; combine with faster indicators for timely signals.
- close_200_sma: 200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and identify golden/death cross setups. Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries.
- close_10_ema: 10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum and potential entry points. Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals.

MACD Related:
- macd: MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and divergence as signals of trend changes. Tips: Confirm with other indicators in low-volatility or sideways markets.
- macds: MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line to trigger trades. Tips: Should be part of a broader strategy to avoid false positives.
- macdh: MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize momentum strength and spot divergence early. Tips: Can be volatile; complement with additional filters in fast-moving markets.

Momentum Indicators:
- rsi: RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis.

Volatility Indicators:
- boll: Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts as a dynamic benchmark for price movement. Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals.
- boll_ub: Bollinger Upper Band: Typically 2 standard deviations above the middle line. Usage: Signals potential overbought conditions and breakout zones. Tips: Confirm signals with other tools; prices may ride the band in strong trends.
- boll_lb: Bollinger Lower Band: Typically 2 standard deviations below the middle line. Usage: Indicates potential oversold conditions. Tips: Use additional analysis to avoid false reversal signals.
- atr: ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust position sizes based on current market volatility. Tips: It's a reactive measure, so use it as part of a broader risk management strategy.

Volume-Based Indicators:
- vwma: VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price action with volume data. Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses.

- Select indicators that provide diverse and complementary information. Avoid redundancy (e.g., do not select both rsi and stochrsi). Also briefly explain why they are suitable for the given market context. When you tool call, please use the exact name of the indicators provided above as they are defined parameters, otherwise your call will fail. Please make sure to call get_stock_data first to retrieve the CSV that is needed to generate indicators. Then use get_indicators with the specific indicator names.

Before writing the final report, call get_verified_market_snapshot for this ticker and the current date, and treat it as the source of truth for any exact OHLCV, price-level, or indicator-value claim. If another tool's output conflicts with the verified snapshot, flag the discrepancy rather than inventing a reconciled number. Do not claim historical validation, support/resistance bounces, or exact percentage moves unless they are directly supported by tool output with concrete dates and prices.

Write a very detailed and nuanced report of the trends you observe. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."""
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "market_report": report,
        }

    return market_analyst_node
