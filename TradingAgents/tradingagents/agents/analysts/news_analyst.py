from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_global_news,
    get_instrument_context_from_state,
    get_language_instruction,
    get_macro_indicators,
    get_news,
    get_prediction_markets,
)


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        asset_label = "company" if asset_type == "stock" else "asset"
        instrument_context = get_instrument_context_from_state(state)
        shadow = state.get("shadow_snapshot")

        if shadow is not None:
            # ---- SHADOW MODE: no external news. Stock lane: earnings date +
            # gap/volume from the snapshot. Crypto lane (removed): was the
            # btc-swing events block — no longer used.
            snapshot, symbol = shadow["snapshot"], shadow["symbol"]
            if shadow.get("kind") == "stock":
                l = snapshot.get("levels") or {}
                s = snapshot.get("session") or {}
                block = (
                    "## Scheduled events and flow (from the stock snapshot; no other news sources)\n"
                    f"- earnings date: {snapshot.get('earnings_date', 'n/a')}\n"
                    f"- 5d return: {l.get('ret_5d')}% | trend: {snapshot.get('trend', 'n/a')}\n"
                    f"- 20d avg volume: {l.get('avg_vol_20d')}\n"
                    f"- session volume pace: {s.get('vol_pace', 'n/a')}x\n"
                    f"- gap vs prior close: {s.get('gap_pct', 'n/a')}%\n"
                )
                system_message = (
                    "You are the News Analyst for a US equity watchlist (NVDA/TSLA/AAPL/AMD/SPY). "
                    "Your ENTIRE input is the scheduled-events block below, taken from the verified "
                    "stock snapshot. Report factually what is scheduled (earnings proximity, event "
                    "risk) and what the flow figures show (gap, volume pace, trend).\n"
                    "RULES (hard):\n"
                    "1. Cite ONLY figures present in the block. Do not fetch news, do not add "
                    "external knowledge, do not speculate about unlisted events.\n"
                    "2. Flag earnings proximity and gap/volume context factually; do not make a "
                    "directional prediction.\n"
                    "3. If the block lacks a figure, say 'not in snapshot'.\n"
                    "Write a concise factual report with a Markdown table of the key fields at the end."
                    + "\n\n" + block + "\n" + get_language_instruction()
                )
            else:
                a = snapshot.get("assets", {}).get(symbol) or {}
                block = (
                    "## Scheduled events and flow (from the btc-swing snapshot; no other news sources)\n"
                    f"- next events: {snapshot.get('events_line', '—')}\n"
                    f"- next_event: {snapshot.get('next_event')} at {snapshot.get('next_event_time')} "
                    f"(in {snapshot.get('next_event_in_h')}h); second: {snapshot.get('next_event_2')} "
                    f"at {snapshot.get('next_event_2_time')} (in {snapshot.get('next_event_2_in_h')}h)\n"
                    f"- event blackout: {snapshot.get('event_blackout')} ({snapshot.get('blackout_reason') or 'none'})\n"
                    f"- ETF flows: {snapshot.get('etf_flow_line', '—')}\n"
                    f"- macro indicators: dxy {snapshot.get('dxy_pct')}% | nasdaq {snapshot.get('nasdaq_pct')}% | regime {snapshot.get('macro')}\n"
                    f"- recent releases: {snapshot.get('last_release_line', '—')}\n"
                )
                system_message = (
                    "You are the News Analyst for a deterministic breakout system (btc-swing). "
                    "Your ENTIRE input is the scheduled-events and flow block below, taken from the "
                    "verified snapshot. Report what is scheduled and what the flow figures show "
                    "(ETF net flows, macro drift), factually.\n"
                    "RULES (hard):\n"
                    "1. Cite ONLY figures present in the block. Do not fetch news, do not add external "
                    "knowledge, do not speculate about unlisted events.\n"
                    "2. Flag event-blackout proximity and macro regime factually; do not make a "
                    "directional prediction.\n"
                    "3. If the block lacks a figure, say 'not in snapshot'.\n"
                    "Write a concise factual report with a Markdown table of the key fields at the end."
                    + "\n\n" + block + "\n" + get_language_instruction()
                )
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system",
                     "You are a helpful AI assistant, collaborating with other assistants. "
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
            return {"messages": [result], "news_report": report}

        tools = [
            get_news,
            get_global_news,
            get_macro_indicators,
            get_prediction_markets,
        ]

        system_message = (
            f"You are a news researcher tasked with analyzing recent news and trends over the past week. Please write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. Use the available tools: get_news(ticker, start_date, end_date) for {asset_label}-specific news by ticker symbol, get_global_news(curr_date, look_back_days, limit) for broader macroeconomic news, get_macro_indicators(indicator, curr_date, look_back_days) to ground macro commentary in actual data from FRED (e.g. 'cpi', 'core_pce', 'unemployment', 'fed_funds_rate', '10y_treasury', 'yield_curve'), and get_prediction_markets(topic, limit) for live market-implied probabilities of forward-looking events (e.g. 'Fed rate cut', 'recession 2026', geopolitical or sector events). Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
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
            "news_report": report,
        }

    return news_analyst_node
