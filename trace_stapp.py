import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import json
from datetime import datetime

# Initialize session state
if 'selected_experiment' not in st.session_state:
    st.session_state.selected_experiment = None


def load_experiments():
    storage_dir = Path("traces")
    return [d.name for d in storage_dir.iterdir() if d.is_dir()]


def load_traces(experiment):
    if not experiment:
        return []
    traces = []
    exp_dir = Path("traces") / experiment
    for trace_file in exp_dir.glob("*.json"):
        try:
            with open(trace_file) as f:
                traces.append(json.load(f))
        except Exception as e:
            st.error(f"Error loading {trace_file}: {str(e)}")
    return sorted(traces,key=lambda x:x['timestamp'])


def render_message(role: str, content: str, render_system_prompt: bool = True):
    """Render a chat message with light theme styling"""
    if role == "system" and len(content) >= 150 and render_system_prompt:
        with st.popover(" 🔧 System"):
            st.markdown(content)
    elif role == "system" and len(content) < 150 and render_system_prompt:
        st.markdown(f"""
        <div style="padding: 10px; border-radius: 5px; background-color: #f1f5f9; border: 1px solid #cbd5e1; margin: 5px 0;">
            🔧 <b>System:</b><br>{content}
        </div>
        """, unsafe_allow_html=True)
    elif role == "user" and len(content) >= 150:
        with st.popover(" 🔧 User"):
            st.markdown(content)
    elif role == "user" and len(content) < 150:
        st.markdown(f"""
        <div style="padding: 10px; border-radius: 5px; background-color: #e8f4ff; border: 1px solid #bfdbfe; margin: 5px 0;">
            👤 <b>User:</b><br>{content}
        </div>
        """, unsafe_allow_html=True)
    elif role == "assistant" and len(content) >= 350:
        with st.popover(" 🔧 Assistant"):
            st.markdown(content)
    elif role == "assistant" and len(content) < 350:
        st.markdown(f"""
        <div style="padding: 10px; border-radius: 5px; background-color: #f0fdf4; border: 1px solid #bbf7d0; margin: 5px 0;">
            🤖 <b>Assistant:</b><br>{content}
        </div>
        """, unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="GenAI Tracking", layout="wide")

    st.title("🤖 GenAI Interaction Tracking")

    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        experiments = load_experiments()

        # Simple experiment selection without callback
        selected_experiment = st.selectbox(
            "Select Experiment",
            options=experiments if experiments else ['No experiments found'],
            index=0 if experiments else None,
        )

        st.session_state.selected_experiment = selected_experiment

        with st.popover("Advanced Filters ⚙️"):
            st.slider("Min Total Tokens", 0, 1000, 0, key="min_tokens")
            st.slider("Max Duration (s)", 0.0, 5.0, 5.0, key="max_duration")
            st.multiselect("Status", ["success", "error"], default=["success"], key="status_filter")

    # Main content
    if st.session_state.selected_experiment and st.session_state.selected_experiment != 'No experiments found':
        traces = load_traces(st.session_state.selected_experiment)

        if not traces:
            st.warning(f"No traces found for experiment: {st.session_state.selected_experiment}")
            return

        # Summary metrics
        st.markdown("### 📊 Overview")
        metrics_container = st.container()
        with metrics_container:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                with st.popover("Total Interactions 🔍"):
                    st.metric("Count", len(traces))
                    st.markdown("Number of tracked interactions in this experiment")
                    if traces:
                        st.caption(f"First: {traces[0]['timestamp']}")
                        st.caption(f"Last: {traces[-1]['timestamp']}")

            with col2:
                success_rate = sum(1 for t in traces if t.get("status") == "success") / len(traces) if traces else 0
                with st.popover("Success Rate 📈"):
                    st.metric("Rate", f"{success_rate:.1%}")
                    success_count = sum(1 for t in traces if t.get("status") == "success")
                    error_count = len(traces) - success_count
                    st.markdown(f"✅ Successful: {success_count}")
                    st.markdown(f"❌ Failed: {error_count}")

            with col3:
                avg_duration = sum(t.get("duration", 0) for t in traces) / len(traces) if traces else 0
                with st.popover("Response Times ⚡"):
                    st.metric("Average", f"{avg_duration:.2f}s")
                    if traces:
                        durations = [t.get("duration", 0) for t in traces]
                        st.markdown(f"⚡ Fastest: {min(durations):.2f}s")
                        st.markdown(f"🐌 Slowest: {max(durations):.2f}s")

            with col4:
                total_tokens = sum(
                    t.get('outputs', {}).get('response', {}).get('usage', {}).get('total_tokens', 0)
                    for t in traces
                )
                with st.popover("Token Usage 🎯"):
                    st.metric("Total", total_tokens)
                    if traces:
                        avg_tokens = total_tokens / len(traces)
                        st.markdown(f"Average per call: {avg_tokens:.1f}")

        # Conversations
        st.markdown("### 💬 Conversations")

        # Filter traces based on selected criteria
        min_tokens = st.session_state.get("min_tokens", 0)
        max_duration = st.session_state.get("max_duration", 5.0)
        status_filter = st.session_state.get("status_filter", ["success"])

        # filtered_traces = [
        #     t for t in traces
        #     if (t.get('outputs', {}).get('response', {}).get('usage', {}).get('total_tokens', 0) >= min_tokens and
        #         t.get('duration', 0) <= max_duration and
        #         t.get('status') in status_filter)
        # ]

        filtered_traces = list(traces)

        for trace in filtered_traces:
            with st.expander(f"{trace['trace_id']}", expanded=False):
                col1, col2 = st.columns([3, 1])

                with col1:
                    messages = trace.get('inputs', {}).get('messages', [])
                    sp = {}
                    um = {}
                    convs = []
                    for i, m in enumerate(messages):
                        if m.get("role") == "system":
                            sp = m
                        elif i==(len(messages)-1):
                            um = m
                        else:
                            convs.append(m)

                    messages = []
                    if sp:
                        messages.append(sp)
                    messages.append(um)

                    with st.popover("Conversation"):
                        for m in convs:
                            render_message(m.get('role'), m.get('content'))

                    for msg in messages:
                        render_message(msg.get('role'), msg.get('content'),
                                       render_system_prompt=True)

                    if response_content := trace.get('outputs', {}).get('content'):
                        render_message('assistant', response_content)

                with col2:
                    with st.popover("Interaction Details 📝"):
                        st.markdown("**Timing:**")
                        st.markdown(f"Duration: `{trace.get('duration', 0):.2f}s`")
                        st.markdown(f"Timestamp: `{trace['timestamp']}`")

                        st.markdown("**Model:**")
                        model_info = trace.get('outputs', {}).get('response', {})
                        st.markdown(f"Name: `{model_info.get('model', 'N/A')}`")
                        st.markdown(f"Status: `{trace.get('status', 'N/A')}`")

                        usage = model_info.get('usage', {})
                        if usage:
                            st.markdown("**Tokens:**")
                            st.markdown(f"Prompt: `{usage.get('prompt_tokens', 0)}`")
                            st.markdown(f"Completion: `{usage.get('completion_tokens', 0)}`")
                            st.markdown(f"Total: `{usage.get('total_tokens', 0)}`")

                    with st.popover("Additional Information"):
                        additional_info = trace.get("trace_extra_info", [])
                        for info in additional_info:
                            st.markdown(f"**{info['key']}:** `{info['value']}`")

        # Analysis tabs
        st.markdown("### 📈 Analysis")
        tab1, tab2 = st.tabs(["Response Times", "Token Usage"])

        timeline_data = pd.DataFrame([
            {
                "Trace ID": t["trace_id"],
                "Timestamp": datetime.fromisoformat(t["timestamp"]),
                "Duration": t.get("duration", 0),
                "Tokens": t.get('outputs', {}).get('response', {}).get('usage', {}).get('total_tokens', 0)
            }
            for t in filtered_traces
        ])

        if not timeline_data.empty:
            with tab1:
                with st.popover("About Response Times 📊"):
                    st.markdown("Shows the time taken for each model response")
                    st.markdown("- Higher peaks indicate slower responses")
                    st.markdown("- Look for patterns in response times")

                fig1 = px.line(timeline_data,
                               x="Timestamp",
                               y="Duration",
                               title="Response Times Over Time")
                st.plotly_chart(fig1, use_container_width=True)

            with tab2:
                with st.popover("About Token Usage 📊"):
                    st.markdown("Shows token consumption over time")
                    st.markdown("- Higher bars indicate more tokens used")
                    st.markdown("- Useful for cost estimation")

                fig2 = px.bar(timeline_data,
                              x="Timestamp",
                              y="Tokens",
                              title="Token Usage Over Time")
                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Please select an experiment from the sidebar to view data.")


if __name__ == "__main__":
    main()
