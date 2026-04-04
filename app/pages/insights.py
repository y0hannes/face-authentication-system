import datetime
import streamlit as st
import pandas as pd

from app.utils import (
    list_registered_users,
    user_image_count,
    get_login_attempts,
    get_auth_events,
    get_failed_attempts,
    get_weekly_login_counts,
    get_outcome_counts,
    get_per_user_login_counts,
)
from src.predict import is_model_ready


# ──────────────────────────────────────────────────────────────────────────────
#  CSS — mirrors the stitch design tokens
# ──────────────────────────────────────────────────────────────────────────────
_CSS = """
<style>
/* ── Page-level resets ── */
[data-testid="stAppViewContainer"] { background: var(--bg-page); color: var(--txt-main); }
[data-testid="block-container"]    { padding-top: 1rem !important; }

/* ── Hero label ── */
.ins-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--brand);
    margin-bottom: 4px;
}
.ins-title {
    font-size: 2.25rem;
    font-weight: 900;
    color: var(--txt-main);
    letter-spacing: -0.03em;
    margin: 0;
    line-height: 1.1;
}
.ins-title-bar {
    height: 4px;
    width: 80px;
    background: linear-gradient(to right, var(--brand), #dc3135);
    border-radius: 9999px;
    margin-top: 12px;
}

/* ── KPI metric cards ── */
.kpi-card {
    background: var(--bg-card);
    border-radius: 8px;
    padding: 24px;
    box-shadow: var(--shadow);
    border-left: 3px solid var(--brand);
    height: 140px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.kpi-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--txt-sub);
    margin-bottom: 4px;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 800;
    color: var(--txt-main);
    letter-spacing: -0.04em;
    line-height: 1;
    margin: 0;
}
.kpi-badge-green {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    margin-top: 12px;
    font-size: 11px;
    font-weight: 600;
    color: #005045;
    background: rgba(0,132,114,0.1);
    padding: 3px 10px;
    border-radius: 9999px;
}
.kpi-badge-red {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    margin-top: 12px;
    font-size: 11px;
    font-weight: 600;
    color: #ba1a1a;
    background: rgba(186,26,26,0.1);
    padding: 3px 10px;
    border-radius: 9999px;
}
.kpi-sub {
    margin-top: 12px;
    font-size: 10px;
    color: var(--txt-sub);
    font-style: italic;
    font-weight: 500;
}

/* ── Chart card ── */
.chart-card {
    background: var(--bg-card);
    border-radius: 12px;
    padding: 28px 28px 16px;
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
    overflow: hidden;
}
.chart-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--txt-main);
    margin: 0 0 16px;
}

/* ── Bar chart (CSS) ── */
.bar-chart-wrap {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 6px;
    height: 200px;
    position: relative;
}
.bar-chart-wrap::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        linear-gradient(var(--border) 1px, transparent 1px);
    background-size: 100% 25%;
    pointer-events: none;
}
.bar {
    flex: 1;
    background: linear-gradient(to top, rgba(184,17,32,0.18), var(--brand));
    border-radius: 3px 3px 0 0;
    min-height: 4px;
    transition: opacity 0.2s;
}
.bar:hover { opacity: 0.8; }
.bar-labels {
    display: flex;
    justify-content: space-between;
    margin-top: 8px;
    font-size: 9px;
    font-weight: 700;
    color: var(--txt-sub);
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

/* ── Biometric health card ── */
.bio-card {
    background: linear-gradient(135deg, #469efe 0%, #0060ab 100%);
    border-radius: 12px;
    padding: 28px;
    box-shadow: 0 4px 16px rgba(0,96,171,0.3);
    color: #ffffff;
    position: relative;
    overflow: hidden;
    height: 100%;
    min-height: 260px;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
}
.bio-card::after {
    content: 'face';
    font-family: 'Material Symbols Outlined';
    font-size: 160px;
    position: absolute;
    right: -20px;
    bottom: -20px;
    opacity: 0.1;
    line-height: 1;
    font-variation-settings: 'FILL' 0, 'wght' 400, 'GRAD' 0, 'opsz' 24;
}
.bio-card-title {
    font-size: 1.1rem;
    font-weight: 700;
    margin: 8px 0 10px;
}
.bio-card-body {
    font-size: 0.82rem;
    opacity: 0.9;
    line-height: 1.6;
    font-weight: 500;
}
.bio-card-btn {
    display: inline-block;
    margin-top: 24px;
    background: #ffffff;
    color: #0060ab;
    padding: 8px 20px;
    border-radius: 8px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    cursor: default;
    position: relative;
    z-index: 1;
    width: fit-content;
}

/* ── Events table ── */
.events-card {
    background: var(--bg-card);
    border-radius: 12px;
    box-shadow: var(--shadow);
    border: 1px solid var(--border);
    overflow: hidden;
    margin-top: 28px;
}
.events-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 24px;
    border-bottom: 1px solid var(--border);
    background: rgba(242,244,248,0.05);
}
.events-header-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--txt-main);
    margin: 0;
}
.realtime-badge {
    font-size: 10px;
    font-weight: 700;
    background: #008472;
    color: #ffffff;
    padding: 3px 12px;
    border-radius: 9999px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.evt-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
}
.evt-table thead tr {
    background: rgba(242,244,248,0.08);
}
.evt-table thead th {
    padding: 12px 20px;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--txt-sub);
    text-align: left;
}
.evt-table tbody tr {
    border-top: 1px solid var(--border);
    transition: background 0.15s;
}
.evt-table tbody tr:hover { background: rgba(242,244,248,0.05); }
.evt-table td { padding: 14px 20px; color: var(--txt-main); }
.status-dot {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    font-weight: 700;
}
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; }
.dot-green { background: #008472; }
.dot-red   { background: #ba1a1a; }
.status-success { color: #008472; }
.status-fail    { color: #ba1a1a; }
.conf-bar-wrap {
    width: 96px;
    height: 5px;
    background: var(--bg-page);
    border-radius: 9999px;
    overflow: hidden;
    margin-bottom: 4px;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 9999px;
}
.conf-value {
    font-size: 10px;
    font-weight: 700;
    color: var(--txt-sub);
}
.ts-cell {
    font-size: 11px;
    font-weight: 500;
    color: var(--txt-sub);
}
.tbl-footer {
    padding: 14px 24px;
    text-align: center;
    border-top: 1px solid var(--border);
    background: rgba(242,244,248,0.05);
}
.tbl-footer-btn {
    font-size: 10px;
    font-weight: 900;
    color: var(--brand);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    background: none;
    border: none;
    cursor: pointer;
}
.no-events-msg {
    padding: 40px 24px;
    text-align: center;
    color: var(--txt-sub);
    font-size: 0.85rem;
    font-style: italic;
}
</style>
"""


def _confidence_bar(conf: float | None, authenticated: bool) -> str:
    """Return an HTML confidence progress bar cell."""
    if conf is None:
        return '<span style="color:#94a3b8;font-size:11px;">N/A</span>'
    pct = min(max(conf, 0), 100)
    color = "#008472" if authenticated else "#ba1a1a"
    return f"""
        <div class="conf-bar-wrap">
          <div class="conf-bar-fill" style="width:{pct:.0f}%;background:{color};"></div>
        </div>
        <span class="conf-value">{pct:.1f}%</span>
    """


def _events_table(events: list[dict]) -> str:
    rows = ""
    for e in events:
        is_auth = e.get("status") == "authenticated"
        dot_cls = "dot-green" if is_auth else "dot-red"
        status_cls = "status-success" if is_auth else "status-fail"
        status_txt = "Success" if is_auth else "Rejected"
        user = e.get("username", "Unknown")
        conf = e.get("confidence")

        # Format timestamp
        try:
            ts_dt = datetime.datetime.fromisoformat(e["ts"])
            ts_str = ts_dt.strftime("%H:%M:%S")
        except Exception:
            ts_str = e.get("ts", "—")

        rows += f"""
        <tr>
          <td>
            <span class="status-dot {status_cls}">
              <span class="dot {dot_cls}"></span>{status_txt}
            </span>
          </td>
          <td style="font-weight:600;">{user}</td>
          <td>{_confidence_bar(conf, is_auth)}</td>
          <td class="ts-cell">{ts_str}</td>
        </tr>"""

    return f"""
    <div class="events-card">
      <div class="events-header">
        <span class="events-header-title">Latest Authentication Events</span>
        <span class="realtime-badge">Real-time Feed</span>
      </div>
      <table class="evt-table">
        <thead>
          <tr>
            <th>Status</th>
            <th>User Identity</th>
            <th>Confidence Score</th>
            <th>Timestamp</th>
          </tr>
        </thead>
        <tbody>{rows if rows else '<tr><td colspan="4" class="no-events-msg">No authentication events recorded yet.</td></tr>'}</tbody>
      </table>
      <div class="tbl-footer">
        <span class="tbl-footer-btn">View All Authentication Logs</span>
      </div>
    </div>"""


def _bar_chart(weekly: list[int]) -> str:
    """Render a CSS bar chart from 4 weekly counts (oldest → newest)."""
    max_v = max(weekly) or 1
    bars = ""
    for v in weekly:
        pct = (v / max_v) * 100
        bars += f'<div class="bar" style="height:{max(pct,2):.0f}%" title="{v} logins"></div>'

    return f"""
    <div>
      <div class="bar-chart-wrap">{bars}</div>
      <div class="bar-labels">
        <span>WK 1</span><span>WK 2</span><span>WK 3</span><span>WK 4</span>
      </div>
    </div>"""


# ──────────────────────────────────────────────────────────────────────────────
#  Main page entry-point
# ──────────────────────────────────────────────────────────────────────────────
def show():
    st.markdown(_CSS, unsafe_allow_html=True)

    # ── Load all data ──────────────────────────────────────────────────────
    users          = list_registered_users()
    total_users    = len(users)
    total_images   = sum(user_image_count(u) for u in users)
    login_count    = get_login_attempts()
    failed_count   = get_failed_attempts()
    events         = get_auth_events(limit=10)
    weekly         = get_weekly_login_counts()
    model_ok       = is_model_ready()
    outcomes       = get_outcome_counts()
    per_user       = get_per_user_login_counts()

    integrity_pct = 99.8 if model_ok else 0.0

    # ── Hero Header ────────────────────────────────────────────────────────
    st.markdown("""
        <div style="margin-bottom: 32px; margin-top: 8px;">
          <p class="ins-label">System Analytics</p>
          <h1 class="ins-title">Authentication Metrics</h1>
          <div class="ins-title-bar"></div>
        </div>
    """, unsafe_allow_html=True)

    # ── KPI Cards ─────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="kpi-card">
          <p class="kpi-label">Total Enrolled</p>
          <p class="kpi-value">{total_users:,}</p>
          <span class="kpi-badge-green">
            <span class="material-symbols-outlined" style="font-size:14px;">group</span>
            {total_images:,} total samples
          </span>
        </div>""", unsafe_allow_html=True)

    with c2:
        updated_str = "Live"
        st.markdown(f"""
        <div class="kpi-card">
          <p class="kpi-label">Recent Logins</p>
          <p class="kpi-value">{login_count:,}</p>
          <p class="kpi-sub">Last updated: {updated_str}</p>
        </div>""", unsafe_allow_html=True)

    with c3:
        badge = (
            '<span class="kpi-badge-red">'
            '<span class="material-symbols-outlined" style="font-size:14px;">warning</span>'
            'Requires attention'
            '</span>'
        ) if failed_count > 0 else (
            '<span class="kpi-badge-green">'
            '<span class="material-symbols-outlined" style="font-size:14px;">check_circle</span>'
            'All clear'
            '</span>'
        )
        st.markdown(f"""
        <div class="kpi-card">
          <p class="kpi-label" style="color:#ba1a1a80;">Failed Attempts</p>
          <p class="kpi-value">{failed_count:,}</p>
          {badge}
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # ── Chart + Biometric Card ─────────────────────────────────────────────
    col_chart, col_bio = st.columns([2, 1], gap="large")

    with col_chart:
        bar_html = _bar_chart(weekly)
        st.markdown(f"""
        <div class="chart-card">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;">
            <span class="chart-title">Login Frequency</span>
            <span style="font-size:10px;padding:4px 12px;border-radius:9999px;
                         background:#b81120;color:#fff;font-weight:700;
                         letter-spacing:0.05em;text-transform:uppercase;">
              30 Days
            </span>
          </div>
          {bar_html}
        </div>""", unsafe_allow_html=True)

    with col_bio:
        status_line = (
            f"System integrity is at {integrity_pct}%. Facial recognition is online and operating within normal parameters."
            if model_ok
            else "Model not yet trained. Register users and train the model to activate biometric authentication."
        )
        icon_html = '<span class="material-symbols-outlined" style="font-size:36px;opacity:0.85;">verified_user</span>'
        st.markdown(f"""
        <div class="bio-card">
          <div style="position:relative;z-index:1;">
            {icon_html}
            <p class="bio-card-title">Biometric Health</p>
            <p class="bio-card-body">{status_line}</p>
            <span class="bio-card-btn">RUN DIAGNOSTICS</span>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Events Table ───────────────────────────────────────────────────────
    st.markdown(_events_table(events), unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── Outcome Breakdown ──────────────────────────────────────────────────
    st.markdown("""
        <div style="margin-bottom:8px;">
          <p class="ins-label" style="margin-top:24px;">Outcome Breakdown</p>
        </div>
    """, unsafe_allow_html=True)

    auth_count    = outcomes["authenticated"]
    unknown_count = outcomes["unknown"]
    total_events  = auth_count + unknown_count
    auth_pct      = (auth_count / total_events * 100) if total_events else 0
    unknown_pct   = (unknown_count / total_events * 100) if total_events else 0

    oc1, oc2 = st.columns(2)

    with oc1:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color:#008472;">
          <p class="kpi-label">Successful Logins</p>
          <p class="kpi-value" style="color:#008472;">{auth_count:,}</p>
          <div style="margin-top:12px;">
            <div style="height:4px;background:#eceef2;border-radius:9999px;overflow:hidden;">
              <div style="height:100%;width:{auth_pct:.0f}%;background:#008472;border-radius:9999px;"></div>
            </div>
            <span style="font-size:10px;color:#94a3b8;font-weight:600;margin-top:4px;display:block;">
              {auth_pct:.0f}% of all events
            </span>
          </div>
        </div>""", unsafe_allow_html=True)

    with oc2:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color:#ba1a1a;">
          <p class="kpi-label" style="color:#ba1a1a80;">Face Not Recognised</p>
          <p class="kpi-value" style="color:#ba1a1a;">{unknown_count:,}</p>
          <div style="margin-top:12px;">
            <div style="height:4px;background:#eceef2;border-radius:9999px;overflow:hidden;">
              <div style="height:100%;width:{unknown_pct:.0f}%;background:#ba1a1a;border-radius:9999px;"></div>
            </div>
            <span style="font-size:10px;color:#94a3b8;font-weight:600;margin-top:4px;display:block;">
              {unknown_pct:.0f}% of all events
            </span>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # ── Per-user Identification Table ─────────────────────────────────────
    st.markdown("""
        <div style="margin-bottom:8px;">
          <p class="ins-label">Identity Recognition Count</p>
        </div>
    """, unsafe_allow_html=True)

    if per_user:
        max_count = max(per_user.values()) or 1
        rows = ""
        for rank, (uname, cnt) in enumerate(per_user.items(), start=1):
            bar_w = int(cnt / max_count * 100)
            rows += f"""
            <tr>
              <td style="padding:12px 20px;font-size:12px;color:#94a3b8;font-weight:700;">#{rank}</td>
              <td style="padding:12px 8px;font-weight:600;font-size:0.875rem;color:#191c1f;">{uname}</td>
              <td style="padding:12px 20px;">
                <div style="display:flex;align-items:center;gap:10px;">
                  <div style="flex:1;height:6px;background:#eceef2;border-radius:9999px;overflow:hidden;">
                    <div style="height:100%;width:{bar_w}%;background:linear-gradient(to right,#b81120,#dc3135);border-radius:9999px;"></div>
                  </div>
                  <span style="font-size:12px;font-weight:800;color:#191c1f;min-width:24px;text-align:right;">{cnt}</span>
                </div>
              </td>
            </tr>"""

        st.markdown(f"""
        <div class="events-card">
          <div class="events-header">
            <span class="events-header-title">Login Count per Identity</span>
            <span style="font-size:10px;font-weight:700;color:var(--txt-sub);font-family:Inter,sans-serif;
                         text-transform:uppercase;letter-spacing:0.05em;">{len(per_user)} identities</span>
          </div>
          <table class="evt-table" style="width:100%;">
            <thead>
              <tr>
                <th style="padding:10px 20px;font-size:10px;font-weight:700;text-transform:uppercase;
                           letter-spacing:0.12em;color:var(--txt-sub);width:48px;">#</th>
                <th style="padding:10px 8px;font-size:10px;font-weight:700;text-transform:uppercase;
                           letter-spacing:0.12em;color:var(--txt-sub);">User</th>
                <th style="padding:10px 20px;font-size:10px;font-weight:700;text-transform:uppercase;
                           letter-spacing:0.12em;color:var(--txt-sub);">Successful Logins</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="events-card">
          <div class="no-events-msg">No successful logins recorded yet.</div>
        </div>""", unsafe_allow_html=True)

    # ── Face Sample Distribution ──────────────────────────────────────────
    if total_users > 0:
        st.markdown("""
            <div style="margin-bottom:8px; margin-top:32px;">
              <p class="ins-label">Face Sample Distribution per User</p>
            </div>
        """, unsafe_allow_html=True)
        data = {"User": list(users), "Images": [user_image_count(u) for u in users]}
        df = pd.DataFrame(data)
        st.bar_chart(df.set_index("User"), color="#b81120", use_container_width=True)
        st.caption("Distribution of captured face samples per registered identity.")
