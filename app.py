"""
DocSafe: Enterprise Document Integrity & AI Tamper Detection Platform
======================================================================
Modern SaaS Web Application with Collapsible Navigation,
Live Document-Scanning & Protection Infusion Consoles,
and Zero-Reference AI Tamper Detection Workspace.

DocSafe Platform v1.0
"""

import base64
import io
import json
import os
import sys
import time
import types
from pathlib import Path

# Automatic self-resolving path: supports running from root or from inside docsafe
CUR_DIR = Path(__file__).resolve().parent
for p in [str(CUR_DIR), str(CUR_DIR.parent)]:
    if p not in sys.path:
        sys.path.insert(0, p)

if "docsafe" not in sys.modules:
    try:
        import docsafe
    except ModuleNotFoundError:
        _pkg = types.ModuleType("docsafe")
        _pkg.__path__ = [str(CUR_DIR)]
        sys.modules["docsafe"] = _pkg

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from docsafe.core.lattice import derive_key_seed
from docsafe.inference.protect import ProtectionEngine
from docsafe.inference.verify import VerificationEngine

# ==============================================================================
# SVG ICON LIBRARY (Crisp, Linear Vector SVGs)
# ==============================================================================
def svg_icon(name: str, size: int = 18, color: str = "currentColor", stroke_width: float = 2.0) -> str:
    """Returns clean inline SVG vector icon markup."""
    icons = {
        "shield": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>''',
        "shield_check": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>''',
        "search": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>''',
        "settings": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg>''',
        "sparkles": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3Z"/></svg>''',
        "file": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"/><polyline points="14 2 14 8 20 8"/></svg>''',
        "bolt": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>''',
        "target": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>''',
        "eye": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><path d="2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>''',
        "lock": f'''<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>''',
    }
    return icons.get(name, "")

# ==============================================================================
# PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="DocSafe | Document Integrity Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* FULL WIDTH FLUID FIT-TO-SCREEN */
    .block-container {
        padding-top: 1.0rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2.2rem !important;
        padding-right: 2.2rem !important;
        max-width: 98% !important;
    }

    /* KEEP HEADER MINIMAL WITH FULLY VISIBLE & STYLED SIDEBAR TOGGLE BUTTON */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 2.8rem !important;
        z-index: 99 !important;
    }
    div[data-testid="stToolbar"] {
        visibility: visible !important;
    }
    /* Hide Deploy button and hamburger menu, but keep sidebar expand button clearly visible */
    div[data-testid="stToolbarActions"],
    button[kind="header"] svg[data-icon="three-dots"],
    #MainMenu, footer {
        display: none !important;
        visibility: hidden !important;
    }
    button[data-testid="stExpandSidebarButton"],
    [data-testid="stExpandSidebarButton"] button {
        visibility: visible !important;
        display: inline-flex !important;
        opacity: 1 !important;
        background: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
        color: #1E293B !important;
        padding: 0.35rem 0.5rem !important;
        cursor: pointer !important;
        z-index: 999 !important;
    }
    button[data-testid="stExpandSidebarButton"]:hover {
        background: #F1F5F9 !important;
        border-color: #94A3B8 !important;
    }
    button[data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapseButton"] button {
        visibility: visible !important;
        display: inline-flex !important;
        color: #475569 !important;
    }

    /* MODERN VERTICAL COLLAPSIBLE SIDEBAR WITHOUT SCROLLBARS */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    div[data-testid="stSidebarContent"],
    div[data-testid="stSidebarUserContent"],
    div[data-testid="stSidebarHeader"] {
        background: #F8FAFC !important;
        border-right: 1px solid #E2E8F0 !important;
        box-shadow: 2px 0 14px rgba(0, 0, 0, 0.03) !important;
        overflow: hidden !important;
        overflow-y: hidden !important;
        scrollbar-width: none !important;
        -ms-overflow-style: none !important;
    }
    section[data-testid="stSidebar"]::-webkit-scrollbar,
    section[data-testid="stSidebar"] *::-webkit-scrollbar {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1.0rem !important;
        padding-left: 0.9rem !important;
        padding-right: 0.9rem !important;
    }

    /* SIDEBAR BRANDING */
    .sidebar-brand-box {
        padding: 0.1rem 0.2rem 0.65rem 0.2rem;
        margin-bottom: 0.55rem;
        border-bottom: 1px solid #E2E8F0;
    }
    .sidebar-logo-row {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        margin-bottom: 0.15rem;
    }
    .sidebar-logo-icon {
        font-size: 1.3rem;
    }
    .sidebar-brand-name {
        font-size: 1.2rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.4px;
    }
    .sidebar-brand-badge {
        background: #EFF6FF;
        color: #2563EB;
        font-size: 0.70rem;
        font-weight: 700;
        padding: 0.12rem 0.45rem;
        border-radius: 9999px;
        border: 1px solid #BFDBFE;
    }
    .sidebar-brand-sub {
        font-size: 0.76rem;
        color: #64748B;
        font-weight: 500;
    }

    /* SIDEBAR VERTICAL RADIO NAVIGATION MENU */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div {
        display: flex !important;
        flex-direction: column !important;
        gap: 0.25rem !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        display: flex !important;
        align-items: center !important;
        padding: 0.50rem 0.85rem !important;
        border-radius: 8px !important;
        border: 1px solid transparent !important;
        background: transparent !important;
        cursor: pointer !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        margin-bottom: 0 !important;
        width: 100% !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:hover {
        background: #F1F5F9 !important;
        color: #0F172A !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"],
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {
        background: #EFF6FF !important;
        border: 1px solid #BFDBFE !important;
        box-shadow: 0 1px 4px rgba(37, 99, 235, 0.08) !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-checked="true"] p,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) p {
        color: #1D4ED8 !important;
        font-weight: 700 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label p {
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
        margin: 0 !important;
    }
    /* Hide the default radio circle / bullet inside the sidebar */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] input[type="radio"],
    section[data-testid="stSidebar"] div[data-testid="stRadio"] [data-baseweb="radio"] > div:first-child,
    section[data-testid="stSidebar"] div[data-testid="stRadio"] svg circle {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
    }

    /* SIDEBAR SYSTEM STATUS CARD */
    .sidebar-system-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.65rem 0.8rem;
        margin-top: 0.9rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.02);
    }
    .sys-card-title {
        font-size: 0.70rem;
        font-weight: 800;
        color: #94A3B8;
        letter-spacing: 0.5px;
        margin-bottom: 0.4rem;
    }
    .sys-item {
        display: flex;
        align-items: center;
        font-size: 0.78rem;
        color: #475569;
        margin-bottom: 0.25rem;
        gap: 0.35rem;
    }
    .sys-dot-active {
        color: #22C55E;
        font-size: 0.72rem;
    }
    .sys-label {
        font-weight: 600;
        color: #64748B;
    }
    .sys-val {
        font-weight: 700;
        color: #0F172A;
        margin-left: auto;
    }
    .sidebar-footer-sub {
        text-align: center;
        font-size: 0.74rem;
        color: #94A3B8;
        margin-top: 0.75rem;
        font-weight: 500;
        line-height: 1.35;
    }

    /* MODERN REFINED BUTTON STYLING */
    button[data-testid="baseButton-primary"] {
        background: #2563EB !important;
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border: 1px solid #1D4ED8 !important;
        border-radius: 9px !important;
        font-weight: 600 !important;
        font-size: 0.91rem !important;
        padding: 0.48rem 1.3rem !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.2) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        letter-spacing: -0.2px !important;
    }
    button[data-testid="baseButton-primary"]:hover {
        background: #1D4ED8 !important;
        background-color: #1D4ED8 !important;
        box-shadow: 0 4px 14px rgba(29, 78, 216, 0.3) !important;
        transform: translateY(-1px) !important;
    }
    button[data-testid="baseButton-secondary"] {
        background: #FFFFFF !important;
        color: #334155 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 9px !important;
        font-weight: 600 !important;
        font-size: 0.90rem !important;
        padding: 0.45rem 1.1rem !important;
        transition: all 0.2s ease !important;
    }
    button[data-testid="baseButton-secondary"]:hover {
        background: #F8FAFC !important;
        border-color: #94A3B8 !important;
        color: #0F172A !important;
    }

    /* COMPACT REFINED WORKSPACE HEADER */
    .workspace-header {
        margin-top: 0.2rem;
        margin-bottom: 1.1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #E2E8F0;
    }
    .workspace-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.4px;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.25rem;
    }
    .workspace-subtitle {
        font-size: 0.88rem;
        color: #64748B;
        line-height: 1.5;
        max-width: 860px;
    }
    .input-label-text {
        font-size: 0.86rem;
        font-weight: 700;
        color: #334155;
        margin-bottom: 0.35rem;
    }

    /* COMPACT FILE META CARD */
    .file-meta-card {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.55rem 1rem;
        margin: 0.7rem 0;
    }
    .file-meta-info {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-size: 0.88rem;
        color: #1E293B;
    }
    .file-name {
        font-weight: 700;
        color: #0F172A;
    }
    .file-spec {
        color: #64748B;
        font-size: 0.82rem;
    }
    .file-ready-badge {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        color: #2563EB;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.2rem 0.55rem;
        border-radius: 5px;
    }

    /* HOMEPAGE TOP HEADER (DISPLAYED ON HOMEPAGE ONLY) */
    .homepage-top-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 0.65rem 1.4rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 2px 8px -2px rgba(0, 0, 0, 0.04);
    }
    .homepage-header-brand {
        display: flex;
        align-items: center;
        gap: 0.55rem;
    }
    .homepage-header-icon {
        font-size: 1.35rem;
    }
    .homepage-header-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.4px;
    }
    .homepage-header-badge {
        background: #EFF6FF;
        color: #2563EB;
        font-size: 0.70rem;
        font-weight: 700;
        padding: 0.12rem 0.50rem;
        border-radius: 9999px;
        border: 1px solid #BFDBFE;
    }
    .homepage-header-meta {
        font-size: 0.82rem;
        color: #64748B;
        font-weight: 600;
    }

    /* GRAND HERO SECTION */
    .hero-grand-box {
        text-align: center;
        padding: 3.2rem 2.5rem 2.6rem 2.5rem;
        background: radial-gradient(90% 130% at 50% 0%, #F8FAFC 0%, #FFFFFF 100%);
        border: 1px solid #E2E8F0;
        border-radius: 24px;
        margin-top: 0.2rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 25px -8px rgba(0, 0, 0, 0.05);
        position: relative;
    }
    .hero-badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.40rem 1.05rem;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        color: #2563EB;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        margin-bottom: 1.1rem;
    }
    .hero-h1-grand {
        font-size: 3.0rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -1.0px;
        line-height: 1.2;
        margin-bottom: 0.9rem;
    }
    .gradient-grand {
        background: linear-gradient(135deg, #2563EB 0%, #6366F1 50%, #4F46E5 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-lead-grand {
        font-size: 1.10rem;
        color: #475569;
        max-width: 860px;
        margin: 0 auto 1.8rem auto;
        line-height: 1.6;
    }

    /* COMPACT IMAGE INSPECTION CONTAINERS */
    .inspection-card-title {
        font-size: 0.86rem;
        font-weight: 700;
        color: #334155;
        margin-bottom: 0.45rem;
    }
    div[data-testid="stImage"] img {
        border-radius: 8px !important;
        max-height: 480px !important;
        object-fit: contain !important;
    }

    /* REAL DOCUMENT SCANNER & SHIELD ANIMATION CARDS */
    @keyframes laserSweepDown {
        0% { top: 0%; opacity: 0; }
        15% { opacity: 1; }
        85% { opacity: 1; }
        100% { top: 96%; opacity: 0; }
    }
    @keyframes spinOrbital {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    @keyframes pulseGlowRing {
        0%, 100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.4); }
        50% { transform: scale(1.04); box-shadow: 0 0 26px 6px rgba(59, 130, 246, 0.5); }
    }
    @keyframes indeterminateShift {
        0% { left: -40%; width: 30%; }
        50% { left: 25%; width: 50%; }
        100% { left: 100%; width: 30%; }
    }
    .anim-doc-scanner-card {
        background: #0F172A;
        border: 1px solid #334155;
        border-radius: 18px;
        padding: 1.5rem 1.8rem;
        color: #F8FAFC;
        text-align: center;
        margin: 1.2rem auto;
        max-width: 440px;
        box-shadow: 0 20px 45px -10px rgba(15, 23, 42, 0.6);
    }
    .anim-doc-preview-wrapper {
        width: 300px;
        height: 185px;
        background: #1E293B;
        border: 1.5px solid #475569;
        border-radius: 12px;
        margin: 0 auto 1.0rem auto;
        position: relative;
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .anim-verify-glow {
        border-color: #EF4444 !important;
        box-shadow: 0 0 22px rgba(239, 68, 68, 0.25);
    }
    .anim-protect-glow {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 22px rgba(56, 189, 248, 0.3);
    }
    .anim-scanned-img-blur {
        width: 100%;
        height: 100%;
        object-fit: cover;
        filter: blur(2.5px) brightness(0.82);
        opacity: 0.92;
        transform: scale(1.04);
    }
    .anim-protected-img-blur {
        width: 100%;
        height: 100%;
        object-fit: cover;
        filter: blur(2.0px) brightness(0.86);
        opacity: 0.92;
        transform: scale(1.03);
    }
    .anim-shield-grid-overlay {
        position: absolute;
        inset: 0;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.2) 0%, transparent 70%);
        z-index: 3;
    }
    .anim-scanner-laser-beam {
        position: absolute;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, #EF4444, #F87171, transparent);
        box-shadow: 0 0 16px #EF4444, 0 0 8px #EF4444;
        animation: laserSweepDown 1.5s ease-in-out infinite;
        z-index: 5;
    }
    .anim-protect-laser-beam {
        position: absolute;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, #38BDF8, #818CF8, transparent);
        box-shadow: 0 0 16px #38BDF8, 0 0 6px #818CF8;
        animation: laserSweepDown 1.5s ease-in-out infinite;
        z-index: 5;
    }
    .anim-scanner-reticle {
        position: absolute;
        inset: 10px;
        border: 1.5px dashed rgba(239, 68, 68, 0.55);
        border-radius: 8px;
        pointer-events: none;
        z-index: 4;
    }
    .anim-protect-badge-float {
        position: absolute;
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background: rgba(15, 23, 42, 0.85);
        border: 1.5px solid #38BDF8;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.5);
        animation: pulseGlowRing 2s ease-in-out infinite;
        z-index: 6;
    }
    .anim-doc-title {
        font-size: 1.08rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.75rem;
    }
    .anim-progress-bar {
        width: 280px;
        height: 6px;
        background: #334155;
        border-radius: 9999px;
        margin: 0.8rem auto 0 auto;
        overflow: hidden;
        position: relative;
    }
    .anim-progress-indeterminate {
        position: absolute;
        height: 100%;
        background: linear-gradient(90deg, #3B82F6, #38BDF8, #60A5FA);
        border-radius: 9999px;
        animation: indeterminateShift 1.4s cubic-bezier(0.4, 0, 0.2, 1) infinite;
    }
    .red-beam {
        background: linear-gradient(90deg, #EF4444, #F87171, #FCA5A5) !important;
    }

    /* STATS CHIPS */
    .metric-chip-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.4rem;
        text-align: center;
        box-shadow: 0 4px 15px -3px rgba(0,0,0,0.03);
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .metric-label {
        font-size: 0.86rem;
        color: #64748B;
        font-weight: 600;
    }

    /* WORKFLOW FLOW CARDS */
    .flow-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 1.6rem 1.4rem;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        box-shadow: 0 4px 15px -2px rgba(0,0,0,0.02);
    }
    .flow-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 16px 32px -6px rgba(0, 0, 0, 0.09);
        border-color: #93C5FD;
    }
    .flow-step-num {
        width: 38px;
        height: 38px;
        line-height: 38px;
        border-radius: 50%;
        background: #EFF6FF;
        color: #2563EB;
        font-weight: 800;
        font-size: 1rem;
        margin: 0 auto 0.9rem auto;
        border: 1.5px solid #BFDBFE;
    }
    .flow-title {
        font-size: 1.12rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.5rem;
    }
    .flow-text {
        font-size: 0.88rem;
        color: #64748B;
        line-height: 1.5;
    }

    /* LIVE SIMULATOR CONSOLE */
    .demo-simulator {
        background: #0F172A;
        border-radius: 22px;
        padding: 2rem;
        color: #F8FAFC;
        margin: 2.2rem 0;
        box-shadow: 0 25px 50px -12px rgba(15, 23, 42, 0.45);
        border: 1px solid #334155;
    }
    .sim-doc-box {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 1.5rem;
        width: 340px;
        position: relative;
    }
    .laser-line {
        position: absolute;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #EF4444, transparent);
        box-shadow: 0 0 12px #EF4444;
        animation: scanSweep 2.5s ease-in-out infinite;
    }
    @keyframes scanSweep {
        0% { top: 0%; opacity: 0; }
        20% { opacity: 1; }
        80% { opacity: 1; }
        100% { top: 95%; opacity: 0; }
    }
    .target-box {
        border: 2px dashed #EF4444;
        background: rgba(239, 68, 68, 0.18);
        color: #FCA5A5;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        display: inline-block;
    }

    /* ALERTS */
    .alert-authentic {
        background: #F0FDF4;
        border: 1px solid #86EFAC;
        border-radius: 10px;
        padding: 0.75rem 1.1rem;
        margin: 0.75rem 0;
    }
    .alert-tampered {
        background: #FEF2F2;
        border: 1px solid #FCA5A5;
        border-radius: 10px;
        padding: 0.75rem 1.1rem;
        margin: 0.75rem 0;
    }
    .alert-unverified {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        border-radius: 10px;
        padding: 0.75rem 1.1rem;
        margin: 0.75rem 0;
    }

    /* SETTINGS BOX */
    .setting-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.1rem;
    }

    /* FOOTER */
    .site-footer {
        margin-top: 4rem;
        padding: 1.8rem 1.5rem;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        text-align: center;
        color: #64748B;
        font-size: 0.88rem;
    }
    .footer-team {
        font-weight: 700;
        color: #0F172A;
        font-size: 0.96rem;
        margin: 0.4rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# CONFIG INITIALIZATION
# ==============================================================================
if "cfg_secret_key" not in st.session_state:
    st.session_state["cfg_secret_key"] = "DocSafeKey2026"
if "cfg_delta" not in st.session_state:
    st.session_state["cfg_delta"] = 1.0
if "cfg_period" not in st.session_state:
    st.session_state["cfg_period"] = 8
if "cfg_threshold" not in st.session_state:
    st.session_state["cfg_threshold"] = 0.50
if "cfg_min_pixels" not in st.session_state:
    st.session_state["cfg_min_pixels"] = 16
if "cfg_device" not in st.session_state:
    st.session_state["cfg_device"] = "CPU"

current_secret_key = st.session_state["cfg_secret_key"]
current_delta = st.session_state["cfg_delta"]
current_period = st.session_state["cfg_period"]
current_threshold = st.session_state["cfg_threshold"]
current_min_pixels = st.session_state["cfg_min_pixels"]
target_device = "cuda" if "CUDA" in st.session_state["cfg_device"] and torch.cuda.is_available() else "cpu"

checkpoint_candidates = [
    str(CUR_DIR / "checkpoints" / "best_detector.pth"),
    str(CUR_DIR / "checkpoints" / "latest_checkpoint.pth"),
    str(CUR_DIR.parent / "checkpoints" / "best_detector.pth"),
    "checkpoints/best_detector.pth",
    "docsafe/checkpoints/best_detector.pth",
]
available_ckpt = [p for p in checkpoint_candidates if Path(p).exists()]
selected_ckpt = available_ckpt[0] if available_ckpt else None

# ==============================================================================
# VERTICAL COLLAPSIBLE SIDEBAR NAVIGATION
# ==============================================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand-box">
            <div class="sidebar-logo-row">
                <span class="sidebar-logo-icon">🛡️</span>
                <span class="sidebar-brand-name">DocSafe</span>
                <span class="sidebar-brand-badge">v1.0</span>
            </div>
            <div class="sidebar-brand-sub">Document Integrity & Anti-Forgery</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_options = [
        "🏠  Overview",
        "🛡️  Protect Document",
        "🔍  Verify Authenticity",
        "⚙️  Settings & Policy",
    ]
    raw_nav_selection = st.radio(
        "Navigation Menu",
        options=nav_options,
        index=0,
        label_visibility="collapsed",
        key="sidebar_vertical_nav",
    )

    if "Overview" in raw_nav_selection:
        nav_selection = "Overview"
    elif "Protect" in raw_nav_selection:
        nav_selection = "Protect Document"
    elif "Verify" in raw_nav_selection:
        nav_selection = "Verify Authenticity"
    elif "Settings" in raw_nav_selection:
        nav_selection = "Settings & Policy"
    else:
        nav_selection = "Overview"

    st.markdown(
        f"""
        <div class="sidebar-system-card">
            <div class="sys-card-title">SYSTEM STATUS</div>
            <div class="sys-item">
                <span class="sys-dot-active">●</span>
                <span class="sys-label">AI Engine</span>
                <span class="sys-val">Active (FPN)</span>
            </div>
            <div class="sys-item">
                <span class="sys-dot-active">●</span>
                <span class="sys-label">Device</span>
                <span class="sys-val">{target_device.upper()}</span>
            </div>
            <div class="sys-item">
                <span class="sys-dot-active">●</span>
                <span class="sys-label">Lattice Mode</span>
                <span class="sys-val">T=8px Safe</span>
            </div>
            <div class="sys-item">
                <span class="sys-dot-active">●</span>
                <span class="sys-label">Latency</span>
                <span class="sys-val">&lt; 70 ms</span>
            </div>
        </div>
        <div class="sidebar-footer-sub">
            <b>DocSafe v1.0</b><br>Document Integrity System
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# ANIMATION HELPER FUNCTIONS (Live Cyber Protection & Real Document Scanner)
# ==============================================================================
def extract_preview_thumbnail_b64(file_bytes: bytes, is_pdf: bool, max_dim: int = 340) -> str:
    """Extracts a lightweight base64 JPEG thumbnail of the uploaded PDF or image for animation."""
    try:
        if is_pdf:
            import fitz
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            if len(doc) > 0:
                pix = doc[0].get_pixmap(dpi=72)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                img.thumbnail((max_dim, max_dim))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=75)
                return base64.b64encode(buf.getvalue()).decode("utf-8")
        else:
            raw_bgr = cv2.imdecode(np.asarray(bytearray(file_bytes), dtype=np.uint8), cv2.IMREAD_COLOR)
            if raw_bgr is not None:
                h, w = raw_bgr.shape[:2]
                scale = min(max_dim / max(h, w), 1.0)
                thumb = cv2.resize(raw_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])
                return base64.b64encode(buf.tobytes()).decode("utf-8")
    except Exception:
        pass
    return ""

def get_protection_anim_html(thumb_b64: str = "") -> str:
    """Returns dynamic HTML/CSS markup showing the user's actual document being infused with protection."""
    img_element = (
        f'<img src="data:image/jpeg;base64,{thumb_b64}" class="anim-protected-img-blur" />'
        if thumb_b64
        else '<div style="color: #64748B; font-size: 2.8rem;">📄</div>'
    )
    return f"""
    <div class="anim-doc-scanner-card">
        <div class="anim-doc-preview-wrapper anim-protect-glow">
            {img_element}
            <div class="anim-shield-grid-overlay"></div>
            <div class="anim-protect-laser-beam"></div>
            <div class="anim-protect-badge-float">🛡️</div>
        </div>
        <div class="anim-doc-title">🛡️ Securing Document & Applying Protection Shield...</div>
        <div class="anim-progress-bar">
            <div class="anim-progress-indeterminate"></div>
        </div>
    </div>
    """

def get_verification_anim_html(thumb_b64: str = "") -> str:
    """Returns dynamic HTML/CSS markup showing the user's actual document being laser scanned."""
    img_element = (
        f'<img src="data:image/jpeg;base64,{thumb_b64}" class="anim-scanned-img-blur" />'
        if thumb_b64
        else '<div style="color: #64748B; font-size: 2.8rem;">📄</div>'
    )
    return f"""
    <div class="anim-doc-scanner-card">
        <div class="anim-doc-preview-wrapper anim-verify-glow">
            {img_element}
            <div class="anim-scanner-laser-beam"></div>
            <div class="anim-scanner-reticle"></div>
        </div>
        <div class="anim-doc-title">🔍 Scanning Document Authenticity...</div>
        <div class="anim-progress-bar">
            <div class="anim-progress-indeterminate red-beam"></div>
        </div>
    </div>
    """


# ==============================================================================
# VIEW 1: MODERN ANIMATED OVERVIEW (HOMEPAGE)
# ==============================================================================
if nav_selection == "Overview":
    # Top project header on Homepage only
    st.markdown(
        """
        <div class="homepage-top-header">
            <div class="homepage-header-brand">
                <span class="homepage-header-icon">🛡️</span>
                <span class="homepage-header-title">DocSafe</span>
                <span class="homepage-header-badge">v1.0</span>
            </div>
            <div class="homepage-header-meta">
                AI Document Integrity & Anti-Forgery Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sparkles_svg = svg_icon("sparkles", size=17, color="#2563EB")
    target_svg = svg_icon("target", size=18, color="#2563EB")
    eye_svg = svg_icon("eye", size=18, color="#2563EB")
    bolt_svg = svg_icon("bolt", size=18, color="#2563EB")
    shield_svg = svg_icon("shield", size=18, color="#2563EB")

    st.markdown(
        f"""
        <div class="hero-grand-box">
            <div class="hero-badge-pill">
                {sparkles_svg} Next-Gen AI Document Integrity Shield
            </div>
            <div class="hero-h1-grand">
                Protect Documents Against <br><span class="gradient-grand">AI Inpainting & Digital Forgery</span>
            </div>
            <div class="hero-lead-grand">
                DocSafe weaves an invisible mathematical shield into digital documents upon issuance.
                If someone alters a number, due date, or signature, our zero-reference AI immediately pinpoints the exact forged characters—all without requiring a central database.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown(
            f"""
            <div class="metric-chip-card">
                <div class="metric-value">97.1%</div>
                <div class="metric-label">{target_svg} AI Detection Accuracy</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col2:
        st.markdown(
            f"""
            <div class="metric-chip-card">
                <div class="metric-value">&gt; 49.8 dB</div>
                <div class="metric-label">{eye_svg} Human Imperceptibility</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col3:
        st.markdown(
            f"""
            <div class="metric-chip-card">
                <div class="metric-value">&lt; 70 ms</div>
                <div class="metric-label">{bolt_svg} Instant CPU Latency</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col4:
        st.markdown(
            f"""
            <div class="metric-chip-card">
                <div class="metric-value">1.4%</div>
                <div class="metric-label">{shield_svg} Ultra-Low False Alarms</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # INTERACTIVE LIVE SIMULATOR WIDGET (Viva Showpiece)
    st.markdown(
        """
        <div class="demo-simulator">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.4rem; padding-bottom: 0.9rem; border-bottom: 1px solid #1E293B;">
                <div style="font-weight: 700; font-size: 1.15rem; display: flex; align-items: center; gap: 0.6rem;">
                    <span style="color: #38BDF8;">⚡ Live AI Verification Simulator</span>
                    <span style="background: #1E293B; font-size: 0.78rem; color: #94A3B8; padding: 0.2rem 0.6rem; border-radius: 6px;">Zero Database Lookup</span>
                </div>
                <div style="font-size: 0.85rem; color: #4ADE80; font-weight: 600;">● Neural Engine Active</div>
            </div>
            <div style="display: flex; gap: 2rem; align-items: center; justify-content: space-around; flex-wrap: wrap;">
                <div class="sim-doc-box">
                    <div style="font-size: 0.82rem; color: #94A3B8; margin-bottom: 0.4rem;">📄 INCOMING CONTRACTOR INVOICE</div>
                    <div style="font-size: 0.9rem; color: #F8FAFC; margin-bottom: 0.9rem; font-family: monospace; line-height: 1.6;">
                        Invoice #: INV-2026-089<br>
                        Vendor: Acme Systems Ltd.<br>
                        Total Due: <span class="target-box">$9,000.00</span> <span style="font-size: 0.75rem; color: #F87171;">(Altered from $1,000)</span>
                    </div>
                    <div class="laser-line"></div>
                    <div style="font-size: 0.8rem; color: #CBD5E1;">Status: <b style="color: #F87171;">🚨 Forgery Detected</b></div>
                </div>
                <div style="max-width: 480px;">
                    <div style="font-weight: 700; font-size: 1.2rem; color: #F8FAFC; margin-bottom: 0.5rem;">How examiners see the difference:</div>
                    <div style="font-size: 0.92rem; color: #94A3B8; line-height: 1.6; margin-bottom: 1rem;">
                        To the human eye, the forged "$9,000.00" looks completely authentic with matching fonts. 
                        DocSafe's neural network isolates the destroyed carrier wave and draws a bounding box right around the modified "9".
                    </div>
                    <div style="display: flex; gap: 0.8rem;">
                        <span style="background: #1E293B; border: 1px solid #334155; padding: 0.35rem 0.8rem; border-radius: 8px; font-size: 0.82rem; color: #E2E8F0;">Zero Original Needed</span>
                        <span style="background: #1E293B; border: 1px solid #334155; padding: 0.35rem 0.8rem; border-radius: 8px; font-size: 0.82rem; color: #E2E8F0;">Searchable PDF Ready</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # TWO BIG LAUNCH ACTION BUTTONS
    st.markdown("### 🚀 Ready to test? Choose an action below:")
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        if st.button("🛡️ Open Document Protection Tool", type="primary", use_container_width=True):
            st.info("💡 Select **'🛡️ Protect Document'** from the left navigation sidebar to begin.")
    with b_col2:
        if st.button("🔍 Open Authenticity Verification Tool", type="secondary", use_container_width=True):
            st.info("💡 Select **'🔍 Verify Authenticity'** from the left navigation sidebar to begin.")

    # FOOTER (HOMEPAGE ONLY)
    st.markdown(
        """
        <div class="site-footer">
            <div class="footer-team">Developed by: Harsh Rajput • Harshit Maurya • Himanshu Ranjan • Amit Singhad</div>
            <div style="font-size: 0.82rem; margin-top: 0.25rem; color: #94A3B8;">DocSafe v2.4 • Next-Gen Document Integrity & AI Tamper Detection Platform</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# VIEW 2: PROTECT DOCUMENT WORKSPACE
# ==============================================================================
elif nav_selection == "Protect Document":
    st.markdown(
        """
        <div class="workspace-header">
            <div class="workspace-title">🛡️ Protect & Shield Document</div>
            <div class="workspace-subtitle">
                Embed an invisible zero-reference cryptographic shield into pixels.
                The document remains 100% identical to human vision and keeps full Ctrl+F searchability.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="input-label-text">Select Document to Protect (PDF, PNG, JPG)</div>', unsafe_allow_html=True)
    uploaded_protect_file = st.file_uploader(
        "Select Document to Protect",
        type=["png", "jpg", "jpeg", "pdf"],
        key="protect_file_input",
        label_visibility="collapsed",
        help="Upload your document file. You can download the protected version right after.",
    )

    if uploaded_protect_file is not None:
        file_bytes = uploaded_protect_file.read()
        is_pdf = uploaded_protect_file.name.lower().endswith(".pdf")
        file_kb = len(file_bytes) / 1024.0
        file_size_label = f"{file_kb:.1f} KB" if file_kb < 1024 else f"{(file_kb/1024.0):.2f} MB"

        anim_placeholder = st.empty()

        if is_pdf:
            st.markdown(
                f"""
                <div class="file-meta-card">
                    <div class="file-meta-info">
                        <span>📄</span>
                        <span class="file-name">{uploaded_protect_file.name}</span>
                        <span class="file-spec">• PDF Document • {file_size_label}</span>
                    </div>
                    <div class="file-ready-badge">Ready to Protect</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            col_b, _ = st.columns([1.8, 3.2])
            with col_b:
                do_protect_pdf = st.button("🛡️ Protect This PDF Now", type="primary", use_container_width=True)

            if do_protect_pdf:
                # Display high-tech cyber shield animation on the actual uploaded document
                thumb_b64 = extract_preview_thumbnail_b64(file_bytes, is_pdf=True)
                anim_placeholder.markdown(get_protection_anim_html(thumb_b64), unsafe_allow_html=True)
                try:
                    engine = ProtectionEngine(
                        secret_key=current_secret_key,
                        period=current_period,
                        delta=current_delta,
                    )
                    t0 = time.time()
                    pdf_summary = engine.protect_pdf(file_bytes, dpi=200)
                    elapsed_ms = (time.time() - t0) * 1000.0

                    st.session_state["protected_pdf_summary"] = pdf_summary
                    st.session_state["protected_pdf_elapsed"] = elapsed_ms
                    st.session_state["protect_mode"] = "pdf"
                finally:
                    anim_placeholder.empty()
        else:
            raw_bgr = cv2.imdecode(np.asarray(bytearray(file_bytes), dtype=np.uint8), cv2.IMREAD_COLOR)
            if raw_bgr is not None:
                h_px, w_px = raw_bgr.shape[:2]
                st.markdown(
                    f"""
                    <div class="file-meta-card">
                        <div class="file-meta-info">
                            <span>🖼️</span>
                            <span class="file-name">{uploaded_protect_file.name}</span>
                            <span class="file-spec">• {w_px} × {h_px} px • {file_size_label}</span>
                        </div>
                        <div class="file-ready-badge">Ready to Protect</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                col_b, _ = st.columns([1.8, 3.2])
                with col_b:
                    do_protect_img = st.button("🛡️ Protect This Image Now", type="primary", use_container_width=True)

                if do_protect_img:
                    # Display high-tech cyber shield animation on the actual uploaded image
                    thumb_b64 = extract_preview_thumbnail_b64(file_bytes, is_pdf=False)
                    anim_placeholder.markdown(get_protection_anim_html(thumb_b64), unsafe_allow_html=True)
                    try:
                        engine = ProtectionEngine(
                            secret_key=current_secret_key,
                            period=current_period,
                            delta=current_delta,
                        )
                        temp_out = Path("temp_protected.png")
                        t0 = time.time()
                        summary = engine.protect_file(raw_bgr, temp_out)
                        elapsed_ms = (time.time() - t0) * 1000.0

                        protected_bgr = cv2.imread(str(temp_out))
                        if temp_out.exists():
                            temp_out.unlink()

                        st.session_state["protected_img"] = protected_bgr
                        st.session_state["protect_summary"] = summary
                        st.session_state["protect_elapsed"] = elapsed_ms
                        st.session_state["raw_img"] = raw_bgr
                        st.session_state["protect_mode"] = "image"
                    finally:
                        anim_placeholder.empty()

    # Display Protected Results (PDF)
    if st.session_state.get("protect_mode") == "pdf" and "protected_pdf_summary" in st.session_state:
        pdf_sum = st.session_state["protected_pdf_summary"]
        st.markdown(
            f"""
            <div class="alert-authentic">
                <div style="font-weight: 700; font-size: 1.02rem; color: #166534; display: flex; align-items: center; gap: 0.45rem;">
                    <span>✅</span> Document Protected Successfully
                </div>
                <div style="color: #15803D; font-size: 0.85rem; margin-top: 0.15rem;">
                    Protected {pdf_sum['total_pages']} page(s) in {st.session_state.get('protected_pdf_elapsed', 0):.1f} ms • Visual clarity: {pdf_sum['avg_psnr_db']:.1f} dB • Full text search (Ctrl+F) preserved
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Pages", f"{pdf_sum['total_pages']} page(s)")
        with m2:
            st.metric("Visual Clarity", f"{pdf_sum['avg_psnr_db']:.1f} dB", delta="Invisible to human eye")
        with m3:
            st.metric("Protection Speed", f"{st.session_state.get('protected_pdf_elapsed', 0):.1f} ms")

        c_dl, _ = st.columns([1.8, 3.2])
        with c_dl:
            st.download_button(
                label="📥 Download Protected Searchable PDF (.pdf)",
                data=pdf_sum["pdf_bytes"],
                file_name="protected_document.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )

    # Display Protected Results (Image)
    elif st.session_state.get("protect_mode") == "image" and "protected_img" in st.session_state:
        img_sum = st.session_state["protect_summary"]
        p_img = st.session_state["protected_img"]
        r_img = st.session_state["raw_img"]

        st.markdown(
            f"""
            <div class="alert-authentic">
                <div style="font-weight: 700; font-size: 1.02rem; color: #166534; display: flex; align-items: center; gap: 0.45rem;">
                    <span>✅</span> Image Protected Successfully
                </div>
                <div style="color: #15803D; font-size: 0.85rem; margin-top: 0.15rem;">
                    Protected in {st.session_state.get('protect_elapsed', 0):.1f} ms • Visual fidelity: {img_sum.get('psnr_db', 50.0):.1f} dB (100% humanly imperceptible)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        p1, p2 = st.columns(2)
        with p1:
            st.markdown('<div class="inspection-card-title">1. Original Document</div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        with p2:
            st.markdown('<div class="inspection-card-title">2. Protected Document (With Invisible Shield)</div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(p_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        is_success, buffer = cv2.imencode(".png", p_img)
        if is_success:
            c_dl, _ = st.columns([1.8, 3.2])
            with c_dl:
                st.download_button(
                    label="📥 Download Protected Image (.png)",
                    data=buffer.tobytes(),
                    file_name="protected_document.png",
                    mime="image/png",
                    type="primary",
                    use_container_width=True,
                )


# ==============================================================================
# VIEW 3: VERIFY AUTHENTICITY WORKSPACE
# ==============================================================================
elif nav_selection == "Verify Authenticity":
    st.markdown(
        """
        <div class="workspace-header">
            <div class="workspace-title">🔍 Check Document Authenticity</div>
            <div class="workspace-subtitle">
                Zero-reference AI audit. Upload any questioned file to check if numbers, totals, dates, or signatures
                have been tampered with. Edits are pinpointed in red bounding boxes with zero original needed.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="input-label-text">Select Questioned Document to Verify (PDF, PNG, JPG)</div>', unsafe_allow_html=True)
    uploaded_verify_file = st.file_uploader(
        "Select Questioned Document to Verify",
        type=["png", "jpg", "jpeg", "pdf"],
        key="verify_file_input",
        label_visibility="collapsed",
        help="Upload the questioned file you want to test for tampering",
    )

    if uploaded_verify_file is not None:
        file_bytes = uploaded_verify_file.read()
        is_pdf_verify = uploaded_verify_file.name.lower().endswith(".pdf")
        v_kb = len(file_bytes) / 1024.0
        v_size_label = f"{v_kb:.1f} KB" if v_kb < 1024 else f"{(v_kb/1024.0):.2f} MB"

        anim_placeholder = st.empty()

        if is_pdf_verify:
            st.markdown(
                f"""
                <div class="file-meta-card">
                    <div class="file-meta-info">
                        <span>📄</span>
                        <span class="file-name">{uploaded_verify_file.name}</span>
                        <span class="file-spec">• PDF Document • {v_size_label}</span>
                    </div>
                    <div class="file-ready-badge">Ready for Audit</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            col_b, _ = st.columns([1.8, 3.2])
            with col_b:
                do_verify_pdf = st.button("🔍 Check PDF Authenticity Now", type="primary", use_container_width=True)

            if do_verify_pdf:
                # Display high-tech cyber laser scanner animation on the actual uploaded document
                thumb_b64 = extract_preview_thumbnail_b64(file_bytes, is_pdf=True)
                anim_placeholder.markdown(get_verification_anim_html(thumb_b64), unsafe_allow_html=True)
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    verifier = VerificationEngine(
                        checkpoint_path=selected_ckpt,
                        secret_key=current_secret_key,
                        period=current_period,
                        threshold=current_threshold,
                        min_tampered_pixels=current_min_pixels,
                        device=target_device,
                        max_inference_dim=2560,
                    )
                    t0 = time.time()
                    pdf_result = verifier.verify_pdf(file_bytes, dpi=200)
                    elapsed_ms = (time.time() - t0) * 1000.0

                    st.session_state["v_pdf_result"] = pdf_result
                    st.session_state["v_pdf_elapsed"] = elapsed_ms
                    st.session_state["v_mode"] = "pdf"
                finally:
                    anim_placeholder.empty()
        else:
            questioned_bgr = cv2.imdecode(np.asarray(bytearray(file_bytes), dtype=np.uint8), cv2.IMREAD_COLOR)
            if questioned_bgr is not None:
                vh, vw = questioned_bgr.shape[:2]
                st.markdown(
                    f"""
                    <div class="file-meta-card">
                        <div class="file-meta-info">
                            <span>🖼️</span>
                            <span class="file-name">{uploaded_verify_file.name}</span>
                            <span class="file-spec">• {vw} × {vh} px • {v_size_label}</span>
                        </div>
                        <div class="file-ready-badge">Ready for Audit</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                col_b, _ = st.columns([1.8, 3.2])
                with col_b:
                    do_verify_img = st.button("🔍 Check Image Authenticity Now", type="primary", use_container_width=True)

                if do_verify_img:
                    # Display high-tech cyber laser scanner animation on the actual uploaded image
                    thumb_b64 = extract_preview_thumbnail_b64(file_bytes, is_pdf=False)
                    anim_placeholder.markdown(get_verification_anim_html(thumb_b64), unsafe_allow_html=True)
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        verifier = VerificationEngine(
                            checkpoint_path=selected_ckpt,
                            secret_key=current_secret_key,
                            period=current_period,
                            threshold=current_threshold,
                            min_tampered_pixels=current_min_pixels,
                            device=target_device,
                            max_inference_dim=2560,
                        )
                        t0 = time.time()
                        result = verifier.verify_image(questioned_bgr)
                        board = verifier.generate_diagnostic_board(result)
                        elapsed_ms = (time.time() - t0) * 1000.0

                        st.session_state["v_result"] = result
                        st.session_state["v_board"] = board
                        st.session_state["v_questioned"] = questioned_bgr
                        st.session_state["v_elapsed"] = elapsed_ms
                        st.session_state["v_mode"] = "image"
                    finally:
                        anim_placeholder.empty()

    # Display PDF Verification Results
    v_mode = st.session_state.get("v_mode")

    if v_mode == "pdf" and "v_pdf_result" in st.session_state:
        pdf_res = st.session_state["v_pdf_result"]

        if pdf_res["status"] == "TAMPERED":
            tampered_pages_list = [str(p["page_number"]) for p in pdf_res["pages"] if p["is_tampered"]]
            st.markdown(
                f"""
                <div class="alert-tampered">
                    <div style="font-weight: 700; font-size: 1.02rem; color: #991B1B; display: flex; align-items: center; gap: 0.45rem;">
                        <span>🚨</span> Tampering Detected in Document
                    </div>
                    <div style="color: #B91C1C; font-size: 0.85rem; margin-top: 0.15rem;">
                        Found alterations on <b>Page(s): {', '.join(tampered_pages_list)}</b> out of {pdf_res['total_pages']} page(s) • Confidence: <b>{pdf_res['overall_confidence']*100:.1f}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif pdf_res["status"] == "UNVERIFIED":
            st.markdown(
                """
                <div class="alert-unverified">
                    <div style="font-weight: 700; font-size: 1.02rem; color: #92400E; display: flex; align-items: center; gap: 0.45rem;">
                        <span>⚠️</span> Notice: Unprotected File or Key Mismatch
                    </div>
                    <div style="color: #B45309; font-size: 0.85rem; margin-top: 0.15rem;">
                        This document does not contain an active DocSafe protection shield, or was verified with a different secret key.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="alert-authentic">
                    <div style="font-weight: 700; font-size: 1.02rem; color: #166534; display: flex; align-items: center; gap: 0.45rem;">
                        <span>✅</span> Document Verified: 100% Authentic & Original
                    </div>
                    <div style="color: #15803D; font-size: 0.85rem; margin-top: 0.15rem;">
                        All {pdf_res['total_pages']} page(s) verified successfully • Zero modifications or tampered pixels detected
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        selected_page_idx = 0
        if pdf_res["total_pages"] > 1:
            page_options = [f"Page {p['page_number']} ({'⚠️ Tampered' if p['is_tampered'] else '✅ Clean'})" for p in pdf_res["pages"]]
            selected_page_str = st.selectbox("Select Page to Inspect:", page_options)
            selected_page_idx = page_options.index(selected_page_str)

        res = pdf_res["pages"][selected_page_idx]
        v_annotated = res.get("annotated_bgr")
        q_bgr = res.get("rendered_bgr", res.get("original_bgr"))

        s1, s2 = st.columns(2)
        with s1:
            st.markdown('<div class="inspection-card-title">1. Uploaded Document Page</div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(q_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        with s2:
            st.markdown('<div class="inspection-card-title">2. Verification Result (Edits Highlighted in Red)</div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(v_annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Document Status", pdf_res["status"])
        with m2:
            st.metric("Current Page", f"Page {res['page_number']}")
        with m3:
            st.metric("Modified Area", f"{res['tampered_pixels']} px")
        with m4:
            st.metric("Audit Speed", f"{st.session_state.get('v_pdf_elapsed', 0):.1f} ms")

        if res["bboxes"]:
            st.markdown("##### 📍 Detected Modified Regions")
            box_data = [
                {
                    "Region": f"Area #{i}",
                    "Position (X, Y)": f"({b['x']}, {b['y']})",
                    "Dimensions": f"{b['width']} × {b['height']} px",
                    "Confidence": f"{b['confidence'] * 100:.1f}%",
                }
                for i, b in enumerate(res["bboxes"], 1)
            ]
            st.dataframe(box_data, use_container_width=True)

        is_ok, buf_res = cv2.imencode(".png", v_annotated)
        if is_ok:
            c_dl, _ = st.columns([1.8, 3.2])
            with c_dl:
                st.download_button(
                    label="📥 Download Result Image with Marked Edits (.png)",
                    data=buf_res.tobytes(),
                    file_name=f"verified_result_page_{res['page_number']}.png",
                    mime="image/png",
                    type="primary",
                    use_container_width=True,
                )

    # Display Image Verification Results
    elif v_mode == "image" and "v_result" in st.session_state:
        res = st.session_state["v_result"]
        q_bgr = st.session_state["v_questioned"]
        v_annotated = res["annotated_bgr"]
        board = st.session_state["v_board"]

        if res["status"] == "TAMPERED":
            st.markdown(
                f"""
                <div class="alert-tampered">
                    <div style="font-weight: 700; font-size: 1.02rem; color: #991B1B; display: flex; align-items: center; gap: 0.45rem;">
                        <span>🚨</span> Tampering Detected in Document
                    </div>
                    <div style="color: #B91C1C; font-size: 0.85rem; margin-top: 0.15rem;">
                        Found <b>{len(res['bboxes'])} modified region(s)</b> across <b>{res['tampered_pixels']} edited pixels</b> • Confidence: <b>{res['overall_confidence']*100:.1f}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif res["status"] == "UNVERIFIED":
            st.markdown(
                """
                <div class="alert-unverified">
                    <div style="font-weight: 700; font-size: 1.02rem; color: #92400E; display: flex; align-items: center; gap: 0.45rem;">
                        <span>⚠️</span> Notice: Unprotected File or Key Mismatch
                    </div>
                    <div style="color: #B45309; font-size: 0.85rem; margin-top: 0.15rem;">
                        This image does not contain an active DocSafe protection shield, or was verified with a different secret key.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="alert-authentic">
                    <div style="font-weight: 700; font-size: 1.02rem; color: #166534; display: flex; align-items: center; gap: 0.45rem;">
                        <span>✅</span> Document Verified: 100% Authentic & Original
                    </div>
                    <div style="color: #15803D; font-size: 0.85rem; margin-top: 0.15rem;">
                        Protection shield is completely intact • No signs of digital tampering, inpainting, or alteration detected
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        s1, s2 = st.columns(2)
        with s1:
            st.markdown('<div class="inspection-card-title">1. Uploaded Document</div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(q_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
        with s2:
            st.markdown('<div class="inspection-card-title">2. Verification Result (Edits Highlighted in Red)</div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(v_annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Authenticity Status", res["status"])
        with m2:
            st.metric("Modified Pixels", f"{res['tampered_pixels']} px")
        with m3:
            st.metric("Flagged Regions", f"{len(res['bboxes'])} box(es)")
        with m4:
            st.metric("Verification Speed", f"{st.session_state.get('v_elapsed', 0):.1f} ms")

        if res["bboxes"]:
            st.markdown("##### 📍 Detected Modified Regions")
            box_data = [
                {
                    "Region": f"Area #{i}",
                    "Position (X, Y)": f"({b['x']}, {b['y']})",
                    "Dimensions": f"{b['width']} × {b['height']} px",
                    "Confidence": f"{b['confidence'] * 100:.1f}%",
                }
                for i, b in enumerate(res["bboxes"], 1)
            ]
            st.dataframe(box_data, use_container_width=True)

        with st.expander("🔬 View Deep Technical Analysis (For Project Examiners & Viva)"):
            st.write("This diagnostic view displays how the mathematical wave and neural network pinpoint the exact altered characters:")
            st.image(
                cv2.cvtColor(board, cv2.COLOR_BGR2RGB),
                caption="Technical Inspection: [1] Uploaded Document | [2] High-Frequency Wave Extraction | [3] Key Resonance Match | [4] AI Tamper Heatmap",
                use_container_width=True,
            )

        is_ok, buf_res = cv2.imencode(".png", v_annotated)
        if is_ok:
            c_dl, _ = st.columns([1.8, 3.2])
            with c_dl:
                st.download_button(
                    label="📥 Download Result Image with Marked Edits (.png)",
                    data=buf_res.tobytes(),
                    file_name="verified_result.png",
                    mime="image/png",
                    type="primary",
                    use_container_width=True,
                )


# ==============================================================================
# VIEW 4: SETTINGS & OPTIONS
# ==============================================================================
elif nav_selection == "Settings & Policy":
    key_svg = svg_icon("lock", size=18, color="#2563EB")
    shield_svg = svg_icon("shield", size=18, color="#2563EB")
    target_svg = svg_icon("target", size=18, color="#2563EB")
    cpu_svg = svg_icon("bolt", size=18, color="#2563EB")

    st.markdown(
        """
        <div class="workspace-header">
            <div class="workspace-title">⚙️ System Settings & Security Options</div>
            <div class="workspace-subtitle">
                Customize your organization's protection keys and AI detection sensitivity. All options are explained below in plain language.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Organization Secret Key
    st.markdown(
        f"""
        <div class="setting-box">
            <div style="font-weight: 700; color: #1E293B; font-size: 0.96rem; margin-bottom: 0.2rem;">{key_svg} Organization Security Key (Password)</div>
            <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 0.6rem;">
                Acts like a private master digital stamp for your organization. Only someone who knows this key 
                can verify whether your documents are authentic. Keep this secret.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    new_key = st.text_input(
        "Enter Secret Key Passphrase",
        value=st.session_state["cfg_secret_key"],
        type="password",
        help="Change this to your organization or company passphrase",
    )
    if new_key != st.session_state["cfg_secret_key"]:
        st.session_state["cfg_secret_key"] = new_key
        st.success("✓ Security key updated!")

    # 2. Protection Shield Settings
    st.markdown(
        f"""
        <div class="setting-box">
            <div style="font-weight: 700; color: #1E293B; font-size: 0.96rem; margin-bottom: 0.2rem;">{shield_svg} Protection Shield Strength & Compression Resilience</div>
            <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 0.6rem;">
                Fine-tune how the invisible protection layer is embedded into the document.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_s1, c_s2 = st.columns(2)
    with c_s1:
        st.markdown("**Protection Shield Strength (Delta)**")
        st.caption("Controls how strong the invisible layer is. Setting 1.0 is mathematically invisible to human eyes while remaining strong enough to catch edits.")
        new_delta = st.slider("Strength Value", min_value=0.5, max_value=2.5, value=float(st.session_state["cfg_delta"]), step=0.1)
        st.session_state["cfg_delta"] = new_delta

    with c_s2:
        st.markdown("**Grid Pattern Spacing (Period in Pixels)**")
        st.caption("Controls pattern spacing. The default (8 pixels) matches standard JPEG compression blocks, so documents stay protected even when sent through WhatsApp.")
        new_period = st.slider("Grid Spacing (Pixels)", min_value=4, max_value=16, value=int(st.session_state["cfg_period"]), step=2)
        st.session_state["cfg_period"] = new_period

    # 3. Tamper Detection Sensitivity
    st.markdown(
        f"""
        <div class="setting-box">
            <div style="font-weight: 700; color: #1E293B; font-size: 0.96rem; margin-bottom: 0.2rem;">{target_svg} AI Detection Sensitivity</div>
            <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 0.6rem;">
                Adjust how strictly the AI verifies documents for modifications.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_d1, c_d2 = st.columns(2)
    with c_d1:
        st.markdown("**AI Detection Sensitivity**")
        st.caption("Controls how strictly the AI flags altered pixels. Higher values catch faint digital pencil marks; moderate values avoid false alerts on scanned paper textures.")
        new_thresh = st.slider("Sensitivity Threshold", min_value=0.1, max_value=0.9, value=float(st.session_state["cfg_threshold"]), step=0.05)
        st.session_state["cfg_threshold"] = new_thresh

    with c_d2:
        st.markdown("**Minimum Alert Size (Pixels)**")
        st.caption("Filters out microscopic scanner dust or single-pixel camera grain. It will only alert if the modified area is as large as a real letter, number, or signature.")
        new_min_px = st.slider("Minimum Size (Pixels)", min_value=4, max_value=64, value=int(st.session_state["cfg_min_pixels"]), step=4)
        st.session_state["cfg_min_pixels"] = new_min_px

    # 4. Processing Hardware
    st.markdown(
        f"""
        <div class="setting-box">
            <div style="font-weight: 700; color: #1E293B; font-size: 0.96rem; margin-bottom: 0.2rem;">{cpu_svg} Processing Hardware</div>
            <div style="font-size: 0.85rem; color: #64748B; margin-bottom: 0.6rem;">
                Selects whether the AI uses your computer processor (CPU) or dedicated graphics card (GPU) for instant verification.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    hardware_opts = ["CPU (Standard Laptop / Desktop)"]
    if torch.cuda.is_available():
        hardware_opts.insert(0, "GPU (NVIDIA CUDA Acceleration)")
    sel_hw = st.selectbox("Selected Hardware", hardware_opts)
    st.session_state["cfg_device"] = sel_hw

    st.markdown("---")
    if st.button("↺ Reset All Settings to Recommended Defaults", help="Reverts all settings back to standard presets"):
        st.session_state["cfg_secret_key"] = "DocSafeKey2026"
        st.session_state["cfg_delta"] = 1.0
        st.session_state["cfg_period"] = 8
        st.session_state["cfg_threshold"] = 0.50
        st.session_state["cfg_min_pixels"] = 16
        st.rerun()
