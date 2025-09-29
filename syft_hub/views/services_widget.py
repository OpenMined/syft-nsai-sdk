"""
Services widget HTML template for the SyftBox NSAI SDK.
"""
import json
import uuid
from typing import List, Optional

def get_services_widget_html(
    services: Optional[List] = None,
    service_type: Optional[str] = None,
    datasite: Optional[str] = None,
    tags: Optional[List[str]] = None,
    max_cost: Optional[float] = None,
    health_check: str = "auto",
    page: int = 1,
    items_per_page: int = 50,
    current_user_email: str = "",
    theme: Optional[str] = None,
) -> str:
    """Generate the services widget HTML for web serving."""
    
    from ..utils.theme import generate_adaptive_css, get_current_theme
    
    container_id = f"syft_services_{uuid.uuid4().hex[:8]}"
    
    # Detect current theme if not provided - force dark for Cursor environment
    if theme is None:
        theme = get_current_theme()
        # Additional check for Cursor/VS Code environment
        import os
        if os.environ.get('VSCODE_PID') or os.environ.get('TERM_PROGRAM') == 'vscode':
            # We're in VS Code/Cursor, prefer dark theme
            theme = 'dark'  # Force dark theme in VS Code/Cursor
        elif theme != 'dark':
            # Try to detect if we should use dark mode from environment
            detected_theme = get_current_theme()
            if detected_theme == 'dark':
                theme = 'dark'
    
    # Set body class based on theme - ensure dark theme is applied
    body_class = 'dark-theme' if theme == 'dark' else ''
    body_data_theme = 'data-theme="dark"' if theme == 'dark' else ''
    
    
    # Generate adaptive CSS for services widget
    adaptive_css = generate_adaptive_css('services-widget')

    # Generate complete HTML with the widget
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SyftBox Services</title>
    {adaptive_css}
    <style>
    /* Use CSS custom properties that inherit from VS Code/Cursor/Jupyter environments */
    body {{
        font-family: var(--vscode-editor-font-family, var(--jp-ui-font-family, -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif));
        margin: 0;
        padding: 16px;
        background: var(--syft-bg-color, #fafafa);
        color: var(--syft-text-color, #333);
        font-size: var(--vscode-editor-font-size, var(--jp-ui-font-size1, 13px));
        line-height: 1.5;
        transition: background-color 0.2s ease, color 0.2s ease;
    }}
    
    /* Ensure widget background is properly transparent for notebook integration */
    .cell-output-ipywidget-background {{
        background-color: transparent !important;
    }}
    
    /* Dark theme base styles - VS Code/Cursor compatible */
    body[data-theme="dark"],
    body.dark-theme,
    body[data-vscode-theme-kind*="dark"],
    body.vscode-dark,
    body.vs-dark {{
        background: var(--syft-bg-color, #1e1e1e) !important;
        color: var(--syft-text-color, #f0f0f0) !important;
    }}
    
    /* Force dark mode on container when body has dark theme */
    body[data-theme="dark"] #{container_id},
    body.dark-theme #{container_id},
    body[data-vscode-theme-kind*="dark"] #{container_id},
    body.vscode-dark #{container_id},
    body.vs-dark #{container_id} {{
        background: #2b2b2b !important;
        color: #f0f0f0 !important;
        border: 1px solid #4a4a4a !important;
    }}
    
    body.dark-theme .header {{
        background: #363636 !important;
        border-bottom: 1px solid #4a4a4a !important;
        color: #f0f0f0 !important;
    }}
    
    body.dark-theme .header h2,
    body.dark-theme .header p {{
        color: #f0f0f0 !important;
    }}
    
    body.dark-theme table {{
        background: #2b2b2b !important;
        border: 1px solid #4a4a4a !important;
    }}
    
    body.dark-theme thead {{
        background: #363636 !important;
        border-bottom: 2px solid #4a4a4a !important;
    }}
    
    body.dark-theme th {{
        color: #f0f0f0 !important;
        border-bottom: 1px solid #4a4a4a !important;
    }}
    
    body.dark-theme tbody tr {{
        background: #2b2b2b !important;
        border-bottom: 1px solid #3a3a3a !important;
    }}
    
    body.dark-theme tbody tr:hover {{
        background: #363636 !important;
    }}
    
    body.dark-theme td {{
        color: #f0f0f0 !important;
        border-bottom: 1px solid #3a3a3a !important;
    }}
    
    body.dark-theme .controls {{
        background: #2b2b2b !important;
        border-bottom: 1px solid #4a4a4a !important;
    }}
    
    body.dark-theme .filter-group label {{
        color: #c0c0c0 !important;
    }}
    
    body.dark-theme .filter-select,
    body.dark-theme .filter-input {{
        background: #363636 !important;
        border: 1px solid #4a4a4a !important;
        color: #f0f0f0 !important;
    }}
    
    body.dark-theme .filter-select:focus,
    body.dark-theme .filter-input:focus {{
        border-color: #007acc !important;
        outline: none !important;
    }}
    
    /* Badge colors for dark theme */
    body.dark-theme .badge-free {{
        background: #0d4f14 !important;
        color: #7bc97f !important;
    }}
    
    body.dark-theme .badge-paid {{
        background: #5a1b20 !important;
        color: #ff7979 !important;
    }}
    
    body.dark-theme .badge-online {{
        background: #0d4f14 !important;
        color: #7bc97f !important;
    }}
    
    body.dark-theme .badge-offline {{
        background: #5a1b20 !important;
        color: #ff7979 !important;
    }}
    
    body.dark-theme .badge-unknown {{
        background: #2a2a2a !important;
        color: #b0b0b0 !important;
    }}
    
    body.dark-theme .tag {{
        background: #363636 !important;
        color: #c0c0c0 !important;
        border: 1px solid #4a4a4a !important;
    }}
    
    body.dark-theme .pagination {{
        background: #363636 !important;
        border-top: 1px solid #4a4a4a !important;
    }}
    
    body.dark-theme .pagination button {{
        background: #2b2b2b !important;
        color: #f0f0f0 !important;
        border: 1px solid #4a4a4a !important;
    }}
    
    body.dark-theme .pagination button:hover {{
        background: #363636 !important;
    }}
    
    body.dark-theme .pagination button:disabled {{
        background: #1e1e1e !important;
        color: #666 !important;
    }}
    
    /* Dark theme overrides for main elements - higher specificity */
    @media (prefers-color-scheme: dark) {{
        body #{container_id} {{
            border: 1px solid #4a4a4a !important;
            background: #2b2b2b !important;
            box-shadow: 0 1px 3px rgba(255,255,255,0.1) !important;
        }}
        .header {{
            background: #363636 !important;
            border-bottom: 1px solid #4a4a4a !important;
        }}
        .header h2 {{
            color: #f0f0f0 !important;
        }}
        .header p {{
            color: #c0c0c0 !important;
        }}
        .controls {{
            background: #2b2b2b !important;
            border-bottom: 1px solid #4a4a4a !important;
        }}
        .filter-group label {{
            color: #c0c0c0 !important;
        }}
        .filter-select, .filter-input {{
            background: #363636 !important;
            border: 1px solid #4a4a4a !important;
            color: #f0f0f0 !important;
        }}
        table {{
            background: #2b2b2b !important;
        }}
        thead {{
            background: #363636 !important;
            border-bottom: 1px solid #4a4a4a !important;
        }}
        th {{
            color: #f0f0f0 !important;
            border-bottom: 1px solid #4a4a4a !important;
        }}
        tbody tr {{
            background: #2b2b2b !important;
            border-bottom: 1px solid #4a4a4a !important;
        }}
        tbody tr:hover {{
            background: #363636 !important;
        }}
        td {{
            color: #f0f0f0 !important;
            border-bottom: 1px solid #4a4a4a !important;
        }}
        .badge-free {{
            background: #0d4f14 !important;
            color: #7bc97f !important;
        }}
        .badge-paid {{
            background: #5a1b20 !important;
            color: #ff7979 !important;
        }}
        .badge-online {{
            background: #0d4f14 !important;
            color: #7bc97f !important;
        }}
        .badge-offline {{
            background: #5a1b20 !important;
            color: #ff7979 !important;
        }}
        .badge-unknown {{
            background: #2a2a2a !important;
            color: #b0b0b0 !important;
        }}
        .tag {{
            background: #363636 !important;
            color: #c0c0c0 !important;
            border: 1px solid #4a4a4a !important;
        }}
        .pagination {{
            background: #363636 !important;
            border-top: 1px solid #4a4a4a !important;
        }}
        .pagination button {{
            background: #2b2b2b !important;
            color: #f0f0f0 !important;
            border: 1px solid #4a4a4a !important;
        }}
        .pagination button:hover:not(:disabled) {{
            background: #363636 !important;
        }}
        .pagination button.active {{
            background: #0066cc !important;
            border-color: #0066cc !important;
        }}
    }}
    
    /* Also apply dark theme when body has dark theme class - comprehensive overrides */
    body[data-theme="dark"] #{container_id},
    body.dark-theme #{container_id} {{
        border: 1px solid #4a4a4a !important;
        background: #2b2b2b !important;
        box-shadow: 0 1px 3px rgba(255,255,255,0.1) !important;
    }}
    body[data-theme="dark"] .header,
    body.dark-theme .header {{
        background: #363636 !important;
        border-bottom: 1px solid #4a4a4a !important;
    }}
    body[data-theme="dark"] .header h2,
    body.dark-theme .header h2 {{
        color: #f0f0f0 !important;
    }}
    body[data-theme="dark"] .header p,
    body.dark-theme .header p {{
        color: #c0c0c0 !important;
    }}
    body[data-theme="dark"] .controls,
    body.dark-theme .controls {{
        background: #2b2b2b !important;
        border-bottom: 1px solid #4a4a4a !important;
    }}
    body[data-theme="dark"] table,
    body.dark-theme table {{
        background: #2b2b2b !important;
    }}
    body[data-theme="dark"] thead,
    body.dark-theme thead {{
        background: #363636 !important;
    }}
    body[data-theme="dark"] tbody tr,
    body.dark-theme tbody tr {{
        background: #2b2b2b !important;
        border-bottom: 1px solid #4a4a4a !important;
    }}
    body[data-theme="dark"] tbody tr:hover,
    body.dark-theme tbody tr:hover {{
        background: #363636 !important;
    }}
    body[data-theme="dark"] td,
    body.dark-theme td {{
        color: #f0f0f0 !important;
        border-bottom: 1px solid #4a4a4a !important;
    }}
    body[data-theme="dark"] th,
    body.dark-theme th {{
        color: #f0f0f0 !important;
        border-bottom: 1px solid #4a4a4a !important;
    }}
    
    /* Light theme base styles (must come before dark theme overrides) */
    #{container_id} {{
        border: 1px solid #e1e1e1;
        border-radius: 8px;
        background: #fff;
        max-width: 100%;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}
    
    /* DARK THEME OVERRIDES - Higher specificity and comes after base styles */
    body[data-theme="dark"],
    body.dark-theme {{
        background: #1e1e1e !important;
        color: #f0f0f0 !important;
    }}
    
    /* Dark theme container - target the widget container directly */
    #{container_id}.dark-theme,
    #{container_id}[data-theme="dark"] {{
        border: 1px solid #4a4a4a !important;
        background: #2b2b2b !important;
        box-shadow: 0 1px 3px rgba(255,255,255,0.1) !important;
        color: #f0f0f0 !important;
    }}
    
    /* Dark theme for form controls */
    #{container_id}.dark-theme .controls input,
    #{container_id}.dark-theme .controls select,
    #{container_id}[data-theme="dark"] .controls input,
    #{container_id}[data-theme="dark"] .controls select {{
        background: #363636 !important;
        color: #f0f0f0 !important;
        border: 1px solid #4a4a4a !important;
    }}
    
    /* Dark theme for quick filter buttons */
    #{container_id}.dark-theme .quick-btn,
    #{container_id}[data-theme="dark"] .quick-btn {{
        background: #363636 !important;
        color: #f0f0f0 !important;
        border: 1px solid #4a4a4a !important;
    }}
    
    #{container_id}.dark-theme .quick-btn:hover,
    #{container_id}[data-theme="dark"] .quick-btn:hover {{
        background: #4a4a4a !important;
    }}
    
    /* Dark theme for copy buttons */
    #{container_id}.dark-theme .copy-btn,
    #{container_id}[data-theme="dark"] .copy-btn {{
        background: #363636 !important;
        color: #f0f0f0 !important;
        border: 1px solid #4a4a4a !important;
    }}
    
    #{container_id}.dark-theme .copy-btn:hover,
    #{container_id}[data-theme="dark"] .copy-btn:hover {{
        background: #4a4a4a !important;
    }}
    
    /* Dark theme for pagination buttons */
    #{container_id}.dark-theme .pagination button,
    #{container_id}[data-theme="dark"] .pagination button {{
        background: #363636 !important;
        color: #f0f0f0 !important;
        border: 1px solid #4a4a4a !important;
    }}
    
    #{container_id}.dark-theme .pagination button:hover:not(:disabled),
    #{container_id}[data-theme="dark"] .pagination button:hover:not(:disabled) {{
        background: #4a4a4a !important;
    }}
    
    #{container_id}.dark-theme .pagination button.active,
    #{container_id}[data-theme="dark"] .pagination button.active {{
        background: #0066cc !important;
        border-color: #0066cc !important;
    }}
    
    /* Dark theme for badges - override light backgrounds */
    #{container_id}.dark-theme .badge-free,
    #{container_id}[data-theme="dark"] .badge-free {{
        background: #0d4f14 !important;
        color: #7bc97f !important;
        border: 1px solid #0d4f14 !important;
    }}
    
    #{container_id}.dark-theme .badge-paid,
    #{container_id}[data-theme="dark"] .badge-paid {{
        background: #5a1b20 !important;
        color: #ff7979 !important;
        border: 1px solid #5a1b20 !important;
    }}
    
    #{container_id}.dark-theme .badge-timeout,
    #{container_id}[data-theme="dark"] .badge-timeout {{
        background: #4a3c00 !important;
        color: #ffa726 !important;
        border: 1px solid #4a3c00 !important;
    }}
    
    /* Dark theme for table elements */
    #{container_id}.dark-theme table,
    #{container_id}[data-theme="dark"] table {{
        background: #2b2b2b !important;
        border: 1px solid #4a4a4a !important;
    }}
    
    #{container_id}.dark-theme thead,
    #{container_id}[data-theme="dark"] thead {{
        background: #363636 !important;
        border-bottom: 2px solid #4a4a4a !important;
    }}
    
    #{container_id}.dark-theme th,
    #{container_id}[data-theme="dark"] th {{
        color: #f0f0f0 !important;
        border-bottom: 1px solid #4a4a4a !important;
    }}
    
    #{container_id}.dark-theme tbody tr,
    #{container_id}[data-theme="dark"] tbody tr {{
        background: #2b2b2b !important;
        border-bottom: 1px solid #3a3a3a !important;
    }}
    
    #{container_id}.dark-theme tbody tr:hover,
    #{container_id}[data-theme="dark"] tbody tr:hover {{
        background: #363636 !important;
    }}
    
    #{container_id}.dark-theme td,
    #{container_id}[data-theme="dark"] td {{
        color: #f0f0f0 !important;
        border-bottom: 1px solid #3a3a3a !important;
    }}
    
    /* Dark theme for header */
    #{container_id}.dark-theme .header,
    #{container_id}[data-theme="dark"] .header {{
        background: #363636 !important;
        border-bottom: 1px solid #4a4a4a !important;
        color: #f0f0f0 !important;
    }}
    
    #{container_id}.dark-theme .header h2,
    #{container_id}.dark-theme .header p,
    #{container_id}[data-theme="dark"] .header h2,
    #{container_id}[data-theme="dark"] .header p {{
        color: #f0f0f0 !important;
    }}
    
    /* Dark theme for controls section (between title and table) */
    #{container_id}.dark-theme .controls,
    #{container_id}[data-theme="dark"] .controls {{
        background: #2b2b2b !important;
        border-bottom: 1px solid #4a4a4a !important;
    }}
    
    #{container_id}.dark-theme .quick-filters,
    #{container_id}[data-theme="dark"] .quick-filters {{
        background: #2b2b2b !important;
        border-bottom: 1px solid #4a4a4a !important;
    }}
    
    /* Dark theme for service type tags (chat/search) */
    #{container_id}.dark-theme .service-type,
    #{container_id}[data-theme="dark"] .service-type {{
        background: #363636 !important;
        color: #f0f0f0 !important;
        border: 1px solid #4a4a4a !important;
    }}
    
    /* Dark theme for general tags */
    #{container_id}.dark-theme .tag,
    #{container_id}[data-theme="dark"] .tag {{
        background: #363636 !important;
        color: #c0c0c0 !important;
        border: 1px solid #4a4a4a !important;
    }}
    
    /* Dark theme for status badges - fix borders */
    #{container_id}.dark-theme .badge-online,
    #{container_id}[data-theme="dark"] .badge-online {{
        background: #0d4f14 !important;
        color: #7bc97f !important;
        border: 1px solid #0d4f14 !important;
    }}
    
    #{container_id}.dark-theme .badge-offline,
    #{container_id}[data-theme="dark"] .badge-offline {{
        background: #5a1b20 !important;
        color: #ff7979 !important;
        border: 1px solid #5a1b20 !important;
    }}
    
    #{container_id}.dark-theme .badge-unknown,
    #{container_id}[data-theme="dark"] .badge-unknown {{
        background: #2a2a2a !important;
        color: #b0b0b0 !important;
        border: 1px solid #2a2a2a !important;
    }}
    
    /* Dark theme for pagination footer */
    #{container_id}.dark-theme .pagination,
    #{container_id}[data-theme="dark"] .pagination {{
        background: #363636 !important;
        border-top: 1px solid #4a4a4a !important;
        color: #f0f0f0 !important;
    }}
    
    #{container_id}.dark-theme .pagination span,
    #{container_id}[data-theme="dark"] .pagination span {{
        color: #f0f0f0 !important;
    }}
    
    /* Dark theme for service type badges (chat/search) */
    #{container_id}.dark-theme .badge-chat,
    #{container_id}[data-theme="dark"] .badge-chat {{
        background: #1e3a5f !important;
        color: #64b5f6 !important;
        border: 1px solid #1e3a5f !important;
    }}
    
    #{container_id}.dark-theme .badge-search,
    #{container_id}[data-theme="dark"] .badge-search {{
        background: #1b5e20 !important;
        color: #81c784 !important;
        border: 1px solid #1b5e20 !important;
    }}
    
    /* Dark theme for config status badges */
    #{container_id}.dark-theme .badge-ready,
    #{container_id}.dark-theme .badge-active,
    #{container_id}.dark-theme .badge-configured,
    #{container_id}[data-theme="dark"] .badge-ready,
    #{container_id}[data-theme="dark"] .badge-active,
    #{container_id}[data-theme="dark"] .badge-configured {{
        background: #0d4f14 !important;
        color: #7bc97f !important;
        border: 1px solid #0d4f14 !important;
    }}
    
    /* Dark theme for inactive config status */
    #{container_id}.dark-theme .badge-inactive,
    #{container_id}.dark-theme .badge-disabled,
    #{container_id}[data-theme="dark"] .badge-inactive,
    #{container_id}[data-theme="dark"] .badge-disabled {{
        background: #2a2a2a !important;
        color: #b0b0b0 !important;
        border: 1px solid #2a2a2a !important;
    }}
    
    html body[data-theme="dark"] .header,
    html body.dark-theme .header {{
        background: #363636 !important;
        border-bottom: 1px solid #4a4a4a !important;
    }}
    
    html body[data-theme="dark"] .header h2,
    html body.dark-theme .header h2 {{
        color: #f0f0f0 !important;
    }}
    
    html body[data-theme="dark"] .header p,
    html body.dark-theme .header p {{
        color: #c0c0c0 !important;
    }}
    
    html body[data-theme="dark"] .controls,
    html body.dark-theme .controls {{
        background: #2b2b2b !important;
        border-bottom: 1px solid #4a4a4a !important;
    }}
    
    html body[data-theme="dark"] table,
    html body.dark-theme table {{
        background: #2b2b2b !important;
        color: #f0f0f0 !important;
    }}
    
    html body[data-theme="dark"] thead,
    html body.dark-theme thead {{
        background: #363636 !important;
    }}
    
    html body[data-theme="dark"] th,
    html body.dark-theme th {{
        color: #f0f0f0 !important;
        border-bottom: 1px solid #4a4a4a !important;
    }}
    
    html body[data-theme="dark"] tbody tr,
    html body.dark-theme tbody tr {{
        background: #2b2b2b !important;
        border-bottom: 1px solid #4a4a4a !important;
    }}
    
    html body[data-theme="dark"] tbody tr:hover,
    html body.dark-theme tbody tr:hover {{
        background: #363636 !important;
    }}
    
    html body[data-theme="dark"] td,
    html body.dark-theme td {{
        color: #f0f0f0 !important;
        border-bottom: 1px solid #4a4a4a !important;
    }}
    
    .header {{
        background: #f7f7f7;
        padding: 16px 20px;
        border-bottom: 1px solid #e1e1e1;
    }}
    
    .header h2 {{
        margin: 0 0 4px 0;
        font-size: 17px;
        font-weight: 500;
        color: #1a1a1a;
        letter-spacing: -0.01em;
    }}
    
    .header p {{
        margin: 0;
        font-size: 11px;
        color: #666;
    }}
    
    .controls {{
        padding: 12px 20px;
        background: #f9f9f9;
        border-bottom: 1px solid #e1e1e1;
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
        align-items: center;
    }}
    
    .controls input, .controls select {{
        padding: 6px 10px;
        border: 1px solid #d0d0d0;
        border-radius: 5px;
        font-size: 11px;
        background: #fff;
        font-family: inherit;
        transition: border-color 0.2s ease;
    }}
    
    .controls input {{
        flex: 1;
        min-width: 200px;
    }}
    
    .controls select {{
        min-width: 120px;
    }}
    
    .controls input:focus, .controls select:focus {{
        outline: none;
        border-color: #007acc;
        box-shadow: 0 0 0 2px rgba(0,122,204,0.1);
    }}
    
    .quick-filters {{
        padding: 10px 20px;
        background: #f4f4f4;
        border-bottom: 1px solid #e1e1e1;
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }}
    
    .quick-btn {{
        padding: 4px 10px;
        border: 1px solid #c0c0c0;
        border-radius: 4px;
        background: #fff;
        color: #333;
        font-size: 11px;
        cursor: pointer;
        text-decoration: none;
        display: inline-block;
        font-family: inherit;
        transition: all 0.15s ease;
    }}
    
    .quick-btn:hover {{
        background: #f0f0f0;
        border-color: #999;
        transform: translateY(-1px);
    }}
    
    .table-container {{
        overflow: auto;
        max-height: 400px;
    }}
    
    table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 11px;
    }}
    
    th {{
        background: #f6f6f6;
        padding: 10px;
        text-align: left;
        border-bottom: 1px solid #e1e1e1;
        font-weight: 500;
        font-size: 11px;
        color: #555;
        position: sticky;
        top: 0;
    }}
    
    /* Dark theme table header override - MUST come after light theme styles */
    #{container_id}.dark-theme th,
    #{container_id}[data-theme="dark"] th {{
        background: #363636 !important;
        color: #f0f0f0 !important;
        border-bottom: 1px solid #4a4a4a !important;
    }}
    
    #{container_id}.dark-theme thead,
    #{container_id}[data-theme="dark"] thead {{
        background: #363636 !important;
        border-bottom: 2px solid #4a4a4a !important;
    }}
    
    td {{
        padding: 10px;
        border-bottom: 1px solid #f0f0f0;
        vertical-align: top;
    }}
    
    tbody tr:hover {{
        background: #f8f8f8;
    }}
    
    tbody tr {{
        cursor: pointer;
        transition: background-color 0.1s ease;
    }}
    
    .badge {{
        display: inline-block;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 10px;
        font-weight: 500;
        margin-right: 3px;
    }}
    
    .badge-chat {{
        background: #e1f5fe;
        color: #0277bd;
        border: 1px solid #81d4fa;
    }}
    
    .badge-search {{
        background: #e8f5e8;
        color: #2e7d32;
        border: 1px solid #a5d6a7;
    }}
    
    .badge-free {{
        background: #e8f5e8;
        color: #2e7d32;
        border: 1px solid #a5d6a7;
    }}
    
    .badge-paid {{
        background: #e3f2fd;
        color: #1976d2;
        border: 1px solid #90caf9;
    }}
    
    .badge-online {{
        background: #e8f5e8;
        color: #2e7d32;
        border: 1px solid #a5d6a7;
    }}
    
    .badge-offline {{
        background: #ffebee;
        color: #c62828;
        border: 1px solid #ef9a9a;
    }}
    
    .badge-timeout {{
        background: #fff8e1;
        color: #f57f17;
        border: 1px solid #fff176;
    }}
    
    .badge-unknown {{
        background: #f5f5f5;
        color: #666;
        border: 1px solid #ccc;
    }}
    
    .tag {{
        background: #f0f0f0;
        color: #555;
        padding: 1px 4px;
        border-radius: 2px;
        font-size: 10px;
        margin-right: 2px;
        margin-bottom: 1px;
        display: inline-block;
    }}
    
    .copy-btn {{
        padding: 3px 8px;
        border: 1px solid #c0c0c0;
        border-radius: 3px;
        background: #fff;
        color: #333;
        font-size: 10px;
        cursor: pointer;
        font-family: inherit;
        margin: 2px 0;
        display: block;
        width: 100%;
        text-align: center;
        transition: all 0.15s ease;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    
    /* Copy column specific styling */
    td .copy-btn {{
        min-width: 50px;
        max-width: 100%;
        margin: 2px 0;
        padding: 4px 6px;
        font-size: 9px;
        line-height: 1.2;
        display: block;
        width: 100%;
        text-align: center;
        position: relative;
        z-index: 1;
    }}
    
    /* Copy button container */
    td:last-child {{
        padding: 8px 6px !important;
        vertical-align: middle;
        min-width: 80px;
        width: 80px;
        position: relative;
    }}
    
    /* Ensure copy buttons are visible */
    tbody td:last-child {{
        background: inherit;
        position: relative;
        z-index: 10;
        overflow: visible;
    }}
    
    /* Make sure buttons don't get cut off */
    .table-container {{
        overflow-x: auto;
        overflow-y: visible;
    }}
    
    /* Prevent text overflow in copy column */
    td:last-child .copy-btn {{
        font-size: 9px;
        line-height: 1.1;
        padding: 3px 4px;
        margin: 1px 0;
        min-height: 18px;
    }}
    
    .copy-btn:hover {{
        background: #f0f0f0;
        border-color: #999;
        transform: translateY(-1px);
    }}
    
    .copy-btn.copied {{
        background: #e8f5e8;
        color: #2e7d32;
        border-color: #a5d6a7;
    }}
    
    .pagination {{
        padding: 12px 20px;
        background: #f7f7f7;
        border-top: 1px solid #e1e1e1;
        text-align: center;
        font-size: 11px;
    }}
    
    .pagination button {{
        padding: 6px 12px;
        border: 1px solid #c0c0c0;
        border-radius: 4px;
        background: #fff;
        color: #333;
        font-size: 11px;
        cursor: pointer;
        margin: 0 3px;
        font-family: inherit;
        transition: all 0.15s ease;
    }}
    
    .pagination button:hover:not(:disabled) {{
        background: #f0f0f0;
        border-color: #999;
        transform: translateY(-1px);
    }}
    
    .pagination button:disabled {{
        opacity: 0.5;
        cursor: not-allowed;
    }}
    
    .truncate {{
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 150px;
    }}
    
    /* COMPREHENSIVE DARK MODE OVERRIDES - Must be last to override light mode */
    body[data-theme="dark"] #{container_id},
    body.dark-theme #{container_id},
    body[data-vscode-theme-kind*="dark"] #{container_id},
    body.vscode-dark #{container_id},
    body.vs-dark #{container_id},
    #{container_id}.dark-theme,
    #{container_id}[data-theme="dark"] {{
        background: #2b2b2b !important;
        color: #f0f0f0 !important;
        border: 1px solid #4a4a4a !important;
    }}
    
    /* Dark mode header */
    body[data-theme="dark"] #{container_id} .header,
    body.dark-theme #{container_id} .header,
    body[data-vscode-theme-kind*="dark"] #{container_id} .header,
    body.vscode-dark #{container_id} .header,
    body.vs-dark #{container_id} .header,
    #{container_id}.dark-theme .header,
    #{container_id}[data-theme="dark"] .header {{
        background: #363636 !important;
        border-bottom: 1px solid #4a4a4a !important;
        color: #f0f0f0 !important;
    }}
    
    body[data-theme="dark"] #{container_id} .header h2,
    body.dark-theme #{container_id} .header h2,
    body[data-vscode-theme-kind*="dark"] #{container_id} .header h2,
    body.vscode-dark #{container_id} .header h2,
    body.vs-dark #{container_id} .header h2,
    #{container_id}.dark-theme .header h2,
    #{container_id}[data-theme="dark"] .header h2 {{
        color: #f0f0f0 !important;
    }}
    
    body[data-theme="dark"] #{container_id} .header p,
    body.dark-theme #{container_id} .header p,
    body[data-vscode-theme-kind*="dark"] #{container_id} .header p,
    body.vscode-dark #{container_id} .header p,
    body.vs-dark #{container_id} .header p,
    #{container_id}.dark-theme .header p,
    #{container_id}[data-theme="dark"] .header p {{
        color: #c0c0c0 !important;
    }}
    
    /* Dark mode controls/search bar */
    body[data-theme="dark"] #{container_id} .controls,
    body.dark-theme #{container_id} .controls,
    body[data-vscode-theme-kind*="dark"] #{container_id} .controls,
    body.vscode-dark #{container_id} .controls,
    body.vs-dark #{container_id} .controls,
    #{container_id}.dark-theme .controls,
    #{container_id}[data-theme="dark"] .controls {{
        background: #2b2b2b !important;
        border-bottom: 1px solid #4a4a4a !important;
    }}
    
    body[data-theme="dark"] #{container_id} .controls input,
    body[data-theme="dark"] #{container_id} .controls select,
    body.dark-theme #{container_id} .controls input,
    body.dark-theme #{container_id} .controls select,
    body[data-vscode-theme-kind*="dark"] #{container_id} .controls input,
    body[data-vscode-theme-kind*="dark"] #{container_id} .controls select,
    body.vscode-dark #{container_id} .controls input,
    body.vscode-dark #{container_id} .controls select,
    body.vs-dark #{container_id} .controls input,
    body.vs-dark #{container_id} .controls select,
    #{container_id}.dark-theme .controls input,
    #{container_id}.dark-theme .controls select,
    #{container_id}[data-theme="dark"] .controls input,
    #{container_id}[data-theme="dark"] .controls select {{
        background: #363636 !important;
        border: 1px solid #4a4a4a !important;
        color: #f0f0f0 !important;
    }}
    
    body[data-theme="dark"] #{container_id} .controls input:focus,
    body[data-theme="dark"] #{container_id} .controls select:focus,
    body.dark-theme #{container_id} .controls input:focus,
    body.dark-theme #{container_id} .controls select:focus,
    body[data-vscode-theme-kind*="dark"] #{container_id} .controls input:focus,
    body[data-vscode-theme-kind*="dark"] #{container_id} .controls select:focus,
    body.vscode-dark #{container_id} .controls input:focus,
    body.vscode-dark #{container_id} .controls select:focus,
    body.vs-dark #{container_id} .controls input:focus,
    body.vs-dark #{container_id} .controls select:focus,
    #{container_id}.dark-theme .controls input:focus,
    #{container_id}.dark-theme .controls select:focus,
    #{container_id}[data-theme="dark"] .controls input:focus,
    #{container_id}[data-theme="dark"] .controls select:focus {{
        border-color: #007acc !important;
        box-shadow: 0 0 0 2px rgba(0,122,204,0.2) !important;
    }}
    
    /* Dark mode quick filters */
    body[data-theme="dark"] #{container_id} .quick-filters,
    body.dark-theme #{container_id} .quick-filters,
    body[data-vscode-theme-kind*="dark"] #{container_id} .quick-filters,
    body.vscode-dark #{container_id} .quick-filters,
    body.vs-dark #{container_id} .quick-filters,
    #{container_id}.dark-theme .quick-filters,
    #{container_id}[data-theme="dark"] .quick-filters {{
        background: #2b2b2b !important;
        border-bottom: 1px solid #4a4a4a !important;
    }}
    
    body[data-theme="dark"] #{container_id} .quick-btn,
    body.dark-theme #{container_id} .quick-btn,
    body[data-vscode-theme-kind*="dark"] #{container_id} .quick-btn,
    body.vscode-dark #{container_id} .quick-btn,
    body.vs-dark #{container_id} .quick-btn,
    #{container_id}.dark-theme .quick-btn,
    #{container_id}[data-theme="dark"] .quick-btn {{
        background: #363636 !important;
        color: #f0f0f0 !important;
        border: 1px solid #4a4a4a !important;
    }}
    
    body[data-theme="dark"] #{container_id} .quick-btn:hover,
    body.dark-theme #{container_id} .quick-btn:hover,
    body[data-vscode-theme-kind*="dark"] #{container_id} .quick-btn:hover,
    body.vscode-dark #{container_id} .quick-btn:hover,
    body.vs-dark #{container_id} .quick-btn:hover,
    #{container_id}.dark-theme .quick-btn:hover,
    #{container_id}[data-theme="dark"] .quick-btn:hover {{
        background: #4a4a4a !important;
    }}
    
    /* Dark mode table */
    body[data-theme="dark"] #{container_id} table,
    body.dark-theme #{container_id} table,
    body[data-vscode-theme-kind*="dark"] #{container_id} table,
    body.vscode-dark #{container_id} table,
    body.vs-dark #{container_id} table,
    #{container_id}.dark-theme table,
    #{container_id}[data-theme="dark"] table {{
        background: #2b2b2b !important;
        color: #f0f0f0 !important;
    }}
    
    /* Dark mode table headers */
    body[data-theme="dark"] #{container_id} th,
    body.dark-theme #{container_id} th,
    body[data-vscode-theme-kind*="dark"] #{container_id} th,
    body.vscode-dark #{container_id} th,
    body.vs-dark #{container_id} th,
    #{container_id}.dark-theme th,
    #{container_id}[data-theme="dark"] th {{
        background: #363636 !important;
        color: #f0f0f0 !important;
        border-bottom: 1px solid #4a4a4a !important;
    }}
    
    body[data-theme="dark"] #{container_id} thead,
    body.dark-theme #{container_id} thead,
    body[data-vscode-theme-kind*="dark"] #{container_id} thead,
    body.vscode-dark #{container_id} thead,
    body.vs-dark #{container_id} thead,
    #{container_id}.dark-theme thead,
    #{container_id}[data-theme="dark"] thead {{
        background: #363636 !important;
        border-bottom: 2px solid #4a4a4a !important;
    }}
    
    /* Dark mode table rows */
    body[data-theme="dark"] #{container_id} tbody tr,
    body.dark-theme #{container_id} tbody tr,
    body[data-vscode-theme-kind*="dark"] #{container_id} tbody tr,
    body.vscode-dark #{container_id} tbody tr,
    body.vs-dark #{container_id} tbody tr,
    #{container_id}.dark-theme tbody tr,
    #{container_id}[data-theme="dark"] tbody tr {{
        background: #2b2b2b !important;
        border-bottom: 1px solid #3a3a3a !important;
    }}
    
    body[data-theme="dark"] #{container_id} tbody tr:hover,
    body.dark-theme #{container_id} tbody tr:hover,
    body[data-vscode-theme-kind*="dark"] #{container_id} tbody tr:hover,
    body.vscode-dark #{container_id} tbody tr:hover,
    body.vs-dark #{container_id} tbody tr:hover,
    #{container_id}.dark-theme tbody tr:hover,
    #{container_id}[data-theme="dark"] tbody tr:hover {{
        background: #363636 !important;
    }}
    
    body[data-theme="dark"] #{container_id} td,
    body.dark-theme #{container_id} td,
    body[data-vscode-theme-kind*="dark"] #{container_id} td,
    body.vscode-dark #{container_id} td,
    body.vs-dark #{container_id} td,
    #{container_id}.dark-theme td,
    #{container_id}[data-theme="dark"] td {{
        color: #f0f0f0 !important;
        border-bottom: 1px solid #3a3a3a !important;
    }}
    
    /* Dark mode pagination */
    body[data-theme="dark"] #{container_id} .pagination,
    body.dark-theme #{container_id} .pagination,
    body[data-vscode-theme-kind*="dark"] #{container_id} .pagination,
    body.vscode-dark #{container_id} .pagination,
    body.vs-dark #{container_id} .pagination,
    #{container_id}.dark-theme .pagination,
    #{container_id}[data-theme="dark"] .pagination {{
        background: #363636 !important;
        border-top: 1px solid #4a4a4a !important;
        color: #f0f0f0 !important;
    }}
    
    body[data-theme="dark"] #{container_id} .pagination button,
    body.dark-theme #{container_id} .pagination button,
    body[data-vscode-theme-kind*="dark"] #{container_id} .pagination button,
    body.vscode-dark #{container_id} .pagination button,
    body.vs-dark #{container_id} .pagination button,
    #{container_id}.dark-theme .pagination button,
    #{container_id}[data-theme="dark"] .pagination button {{
        background: #2b2b2b !important;
        color: #f0f0f0 !important;
        border: 1px solid #4a4a4a !important;
    }}
    
    body[data-theme="dark"] #{container_id} .pagination button:hover:not(:disabled),
    body.dark-theme #{container_id} .pagination button:hover:not(:disabled),
    body[data-vscode-theme-kind*="dark"] #{container_id} .pagination button:hover:not(:disabled),
    body.vscode-dark #{container_id} .pagination button:hover:not(:disabled),
    body.vs-dark #{container_id} .pagination button:hover:not(:disabled),
    #{container_id}.dark-theme .pagination button:hover:not(:disabled),
    #{container_id}[data-theme="dark"] .pagination button:hover:not(:disabled) {{
        background: #363636 !important;
    }}
    
    body[data-theme="dark"] #{container_id} .pagination button:disabled,
    body.dark-theme #{container_id} .pagination button:disabled,
    body[data-vscode-theme-kind*="dark"] #{container_id} .pagination button:disabled,
    body.vscode-dark #{container_id} .pagination button:disabled,
    body.vs-dark #{container_id} .pagination button:disabled,
    #{container_id}.dark-theme .pagination button:disabled,
    #{container_id}[data-theme="dark"] .pagination button:disabled {{
        background: #1e1e1e !important;
        color: #666 !important;
    }}
    
    body[data-theme="dark"] #{container_id} .pagination span,
    body.dark-theme #{container_id} .pagination span,
    body[data-vscode-theme-kind*="dark"] #{container_id} .pagination span,
    body.vscode-dark #{container_id} .pagination span,
    body.vs-dark #{container_id} .pagination span,
    #{container_id}.dark-theme .pagination span,
    #{container_id}[data-theme="dark"] .pagination span {{
        color: #f0f0f0 !important;
    }}
    
    /* Dark mode copy buttons */
    body[data-theme="dark"] #{container_id} .copy-btn,
    body.dark-theme #{container_id} .copy-btn,
    body[data-vscode-theme-kind*="dark"] #{container_id} .copy-btn,
    body.vscode-dark #{container_id} .copy-btn,
    body.vs-dark #{container_id} .copy-btn,
    #{container_id}.dark-theme .copy-btn,
    #{container_id}[data-theme="dark"] .copy-btn {{
        background: #363636 !important;
        color: #f0f0f0 !important;
        border: 1px solid #4a4a4a !important;
        margin: 2px 0;
        padding: 4px 6px;
        font-size: 9px;
        line-height: 1.2;
        min-width: 50px;
        display: block;
        width: 100%;
        text-align: center;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        position: relative;
        z-index: 1;
    }}
    
    /* Dark mode copy column */
    body[data-theme="dark"] #{container_id} td:last-child,
    body.dark-theme #{container_id} td:last-child,
    body[data-vscode-theme-kind*="dark"] #{container_id} td:last-child,
    body.vscode-dark #{container_id} td:last-child,
    body.vs-dark #{container_id} td:last-child,
    #{container_id}.dark-theme td:last-child,
    #{container_id}[data-theme="dark"] td:last-child {{
        padding: 8px 6px !important;
        vertical-align: middle;
        min-width: 80px;
        width: 80px;
        position: relative;
        background: inherit;
        z-index: 10;
        overflow: visible;
    }}
    
    /* Dark mode copy button specific styling */
    body[data-theme="dark"] #{container_id} td:last-child .copy-btn,
    body.dark-theme #{container_id} td:last-child .copy-btn,
    body[data-vscode-theme-kind*="dark"] #{container_id} td:last-child .copy-btn,
    body.vscode-dark #{container_id} td:last-child .copy-btn,
    body.vs-dark #{container_id} td:last-child .copy-btn,
    #{container_id}.dark-theme td:last-child .copy-btn,
    #{container_id}[data-theme="dark"] td:last-child .copy-btn {{
        font-size: 9px;
        line-height: 1.1;
        padding: 3px 4px;
        margin: 1px 0;
        min-height: 18px;
    }}
    
    body[data-theme="dark"] #{container_id} .copy-btn:hover,
    body.dark-theme #{container_id} .copy-btn:hover,
    body[data-vscode-theme-kind*="dark"] #{container_id} .copy-btn:hover,
    body.vscode-dark #{container_id} .copy-btn:hover,
    body.vs-dark #{container_id} .copy-btn:hover,
    #{container_id}.dark-theme .copy-btn:hover,
    #{container_id}[data-theme="dark"] .copy-btn:hover {{
        background: #4a4a4a !important;
    }}
    
    body[data-theme="dark"] #{container_id} .copy-btn.copied,
    body.dark-theme #{container_id} .copy-btn.copied,
    body[data-vscode-theme-kind*="dark"] #{container_id} .copy-btn.copied,
    body.vscode-dark #{container_id} .copy-btn.copied,
    body.vs-dark #{container_id} .copy-btn.copied,
    #{container_id}.dark-theme .copy-btn.copied,
    #{container_id}[data-theme="dark"] .copy-btn.copied {{
        background: #0d4f14 !important;
        color: #7bc97f !important;
        border-color: #0d4f14 !important;
    }}
    
    /* Dark mode badges and tags */
    body[data-theme="dark"] #{container_id} .badge-free,
    body.dark-theme #{container_id} .badge-free,
    body[data-vscode-theme-kind*="dark"] #{container_id} .badge-free,
    body.vscode-dark #{container_id} .badge-free,
    body.vs-dark #{container_id} .badge-free,
    #{container_id}.dark-theme .badge-free,
    #{container_id}[data-theme="dark"] .badge-free {{
        background: #0d4f14 !important;
        color: #7bc97f !important;
        border: 1px solid #0d4f14 !important;
    }}
    
    body[data-theme="dark"] #{container_id} .badge-paid,
    body.dark-theme #{container_id} .badge-paid,
    body[data-vscode-theme-kind*="dark"] #{container_id} .badge-paid,
    body.vscode-dark #{container_id} .badge-paid,
    body.vs-dark #{container_id} .badge-paid,
    #{container_id}.dark-theme .badge-paid,
    #{container_id}[data-theme="dark"] .badge-paid {{
        background: #1e3a8a !important;
        color: #60a5fa !important;
        border: 1px solid #1e3a8a !important;
    }}
    
    body[data-theme="dark"] #{container_id} .badge-online,
    body.dark-theme #{container_id} .badge-online,
    body[data-vscode-theme-kind*="dark"] #{container_id} .badge-online,
    body.vscode-dark #{container_id} .badge-online,
    body.vs-dark #{container_id} .badge-online,
    #{container_id}.dark-theme .badge-online,
    #{container_id}[data-theme="dark"] .badge-online {{
        background: #0d4f14 !important;
        color: #7bc97f !important;
        border: 1px solid #0d4f14 !important;
    }}
    
    body[data-theme="dark"] #{container_id} .badge-offline,
    body.dark-theme #{container_id} .badge-offline,
    body[data-vscode-theme-kind*="dark"] #{container_id} .badge-offline,
    body.vscode-dark #{container_id} .badge-offline,
    body.vs-dark #{container_id} .badge-offline,
    #{container_id}.dark-theme .badge-offline,
    #{container_id}[data-theme="dark"] .badge-offline {{
        background: #5a1b20 !important;
        color: #ff7979 !important;
        border: 1px solid #5a1b20 !important;
    }}
    
    body[data-theme="dark"] #{container_id} .badge-timeout,
    body.dark-theme #{container_id} .badge-timeout,
    body[data-vscode-theme-kind*="dark"] #{container_id} .badge-timeout,
    body.vscode-dark #{container_id} .badge-timeout,
    body.vs-dark #{container_id} .badge-timeout,
    #{container_id}.dark-theme .badge-timeout,
    #{container_id}[data-theme="dark"] .badge-timeout {{
        background: #4a3c00 !important;
        color: #ffa726 !important;
        border: 1px solid #4a3c00 !important;
    }}
    
    body[data-theme="dark"] #{container_id} .badge-unknown,
    body.dark-theme #{container_id} .badge-unknown,
    body[data-vscode-theme-kind*="dark"] #{container_id} .badge-unknown,
    body.vscode-dark #{container_id} .badge-unknown,
    body.vs-dark #{container_id} .badge-unknown,
    #{container_id}.dark-theme .badge-unknown,
    #{container_id}[data-theme="dark"] .badge-unknown {{
        background: #2a2a2a !important;
        color: #b0b0b0 !important;
        border: 1px solid #2a2a2a !important;
    }}
    
    /* Service type badges */
    body[data-theme="dark"] #{container_id} .badge-chat,
    body.dark-theme #{container_id} .badge-chat,
    body[data-vscode-theme-kind*="dark"] #{container_id} .badge-chat,
    body.vscode-dark #{container_id} .badge-chat,
    body.vs-dark #{container_id} .badge-chat,
    #{container_id}.dark-theme .badge-chat,
    #{container_id}[data-theme="dark"] .badge-chat {{
        background: #1e3a5f !important;
        color: #64b5f6 !important;
        border: 1px solid #1e3a5f !important;
    }}
    
    body[data-theme="dark"] #{container_id} .badge-search,
    body.dark-theme #{container_id} .badge-search,
    body[data-vscode-theme-kind*="dark"] #{container_id} .badge-search,
    body.vscode-dark #{container_id} .badge-search,
    body.vs-dark #{container_id} .badge-search,
    #{container_id}.dark-theme .badge-search,
    #{container_id}[data-theme="dark"] .badge-search {{
        background: #1b5e20 !important;
        color: #81c784 !important;
        border: 1px solid #1b5e20 !important;
    }}
    
    /* General tags */
    body[data-theme="dark"] #{container_id} .tag,
    body.dark-theme #{container_id} .tag,
    body[data-vscode-theme-kind*="dark"] #{container_id} .tag,
    body.vscode-dark #{container_id} .tag,
    body.vs-dark #{container_id} .tag,
    #{container_id}.dark-theme .tag,
    #{container_id}[data-theme="dark"] .tag {{
        background: #363636 !important;
        color: #c0c0c0 !important;
        border: 1px solid #4a4a4a !important;
    }}
    </style>
</head>
<body class="{body_class}" {body_data_theme}>
    <div class="syft-widget">
        <div id="{container_id}" class="{'dark-theme' if theme == 'dark' else 'light-theme'}" data-theme="{theme}">
        <div class="header">
            <h2>SyftBox Services</h2>
            <p>Click on any row to see usage examples</p>
        </div>
        
        <div class="controls">
            <input type="text" id="{container_id}-search" placeholder="Search services...">
            <select id="{container_id}-service-type">
                <option value="">All Types</option>
                <option value="chat">Chat</option>
                <option value="search">Search</option>
            </select>
            <select id="{container_id}-pricing">
                <option value="">All Pricing</option>
                <option value="free">Free</option>
                <option value="paid">Paid</option>
            </select>
            <select id="{container_id}-availability">
                <option value="">All Status</option>
                <option value="online">Online</option>
                <option value="offline">Offline</option>
                <option value="timeout">Timeout</option>
            </select>
        </div>
        
        <div class="quick-filters">
            <button class="quick-btn" onclick="quickFilter_{container_id}('free')">Free Only</button>
            <button class="quick-btn" onclick="quickFilter_{container_id}('online')">Online Only</button>
            <button class="quick-btn" onclick="quickFilter_{container_id}('chat')">Chat Services</button>
            <button class="quick-btn" onclick="quickFilter_{container_id}('search')">Search Services</button>
            <button class="quick-btn" onclick="clearFilters_{container_id}()">Clear All</button>
        </div>
        
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th style="width: 14%">Name</th>
                        <th style="width: 14%">Datasite</th>
                        <th style="width: 12%">Type</th>
                        <th style="width: 10%">Pricing</th>
                        <th style="width: 10%">Availability</th>
                        <th style="width: 14%">Tags</th>
                        <th style="width: 16%">Description</th>
                        <th style="width: 10%">Copy</th>
                    </tr>
                </thead>
                <tbody id="{container_id}-tbody">
                    <tr><td colspan="8" style="text-align: center; padding: 20px;">Loading services...</td></tr>
                </tbody>
            </table>
        </div>
        
        <div class="pagination">
            <button onclick="previousPage_{container_id}()" id="{container_id}-prev-btn" disabled>← Previous</button>
            <span id="{container_id}-page-info">Page 1 of 1</span>
            <button onclick="nextPage_{container_id}()" id="{container_id}-next-btn">Next →</button>
        </div>
    </div>

    <script>
    // Simple theme detection for services widget
    function detectAndApplyTheme() {{
        // If body already has dark theme (from server), nothing to do
        if (document.body.hasAttribute('data-theme') && 
            document.body.getAttribute('data-theme') === 'dark') {{
            console.log('Dark theme already applied from server');
            return;
        }}
        
        // Check system preference as fallback
        if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {{
            console.log('Applying dark theme from system preference');
            document.body.setAttribute('data-theme', 'dark');
            document.body.classList.add('dark-theme');
        }}
    }}
    
    // Apply theme detection
    detectAndApplyTheme();
    
    // Listen for changes to system preference
    if (window.matchMedia) {{
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {{
            console.log('System preference changed:', e.matches ? 'dark' : 'light');
            if (e.matches) {{
                document.body.setAttribute('data-theme', 'dark');
                document.body.classList.add('dark-theme');
            }} else {{
                document.body.removeAttribute('data-theme');
                document.body.classList.remove('dark-theme');
            }}
        }});
    }}
    
    (function() {{  // IIFE to isolate this widget instance
    const widgetId = '{container_id}';
    
    // Widget-specific state
    let allServices = [];
    let filteredServices = [];
    let currentPage = 1;
    const itemsPerPage = 20;

    // Initialize services data - json.dumps creates a JavaScript array literal
    const servicesData = {json.dumps(services) if services else '[]'};
    
    // Use services data or demo data
    if (!servicesData || !Array.isArray(servicesData) || servicesData.length === 0) {{
        allServices = [
            {{
                name: "Demo Chat Service",
                datasite: "demo@example.com",
                services: [{{type: "chat", enabled: true}}],
                min_pricing: 0,
                max_pricing: 0,
                config_status: "active",
                health_status: "online",
                tags: ["demo", "chat"],
                summary: "A demo chat service",
                description: "Demo service for testing"
            }},
            {{
                name: "Demo Search Service", 
                datasite: "demo2@example.com",
                services: [{{type: "search", enabled: true}}],
                min_pricing: 0.01,
                max_pricing: 0.05,
                config_status: "active",
                health_status: "offline",
                tags: ["demo", "search"],
                summary: "A demo search service",
                description: "Demo search service"
            }}
        ];
    }} else {{
        allServices = servicesData;
    }}
    
    filteredServices = allServices.slice();

    // Render table
    function renderTable() {{
        const tbody = document.getElementById('{container_id}-tbody');
        
        if (!tbody) {{
            console.error('tbody element not found! Retrying in 100ms...');
            setTimeout(renderTable, 100);
            return;
        }}
        
        const totalPages = Math.ceil(filteredServices.length / itemsPerPage);
        const start = (currentPage - 1) * itemsPerPage;
        const end = Math.min(start + itemsPerPage, filteredServices.length);
        
        if (filteredServices.length === 0) {{
            tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; padding: 20px;">No services found</td></tr>';
        }} else {{
            const rows = filteredServices.slice(start, end).map(service => {{
                const serviceId = service.datasite + '/' + service.name;
                
                // Handle services array properly
                let serviceTypes = [];
                if (service.services && Array.isArray(service.services)) {{
                    serviceTypes = service.services.filter(s => s.enabled);
                }}
                
                const types = serviceTypes.length > 0 ? 
                    serviceTypes.map(s => `<span class="badge badge-${{s.type}}">${{s.type}}</span>`).join('') : 
                    '<span class="badge badge-unknown">none</span>';
                
                const pricing = (service.min_pricing === 0 || service.min_pricing === undefined) ? 
                    '<span class="badge badge-free">Free</span>' : 
                    `<span class="badge badge-paid">$${{service.min_pricing.toFixed(3)}}</span>`;
                
                const status = service.health_status ? 
                    `<span class="badge badge-${{service.health_status}}">${{service.health_status}}</span>` :
                    '<span class="badge badge-unknown">unknown</span>';
                
                const tags = service.tags && service.tags.length > 0 ? 
                    service.tags.slice(0, 3).map(tag => `<span class="tag">${{escapeHtml(tag)}}</span>`).join('') : 
                    '<span class="tag">none</span>';
                
                const moreTagsCount = service.tags && service.tags.length > 3 ? service.tags.length - 3 : 0;
                const tagsDisplay = tags + (moreTagsCount > 0 ? `<span class="tag">+${{moreTagsCount}}</span>` : '');
                
                return `<tr onclick="window['showUsageModal_{container_id}']('${{escapeHtml(service.name)}}', '${{escapeHtml(service.datasite)}}', ${{JSON.stringify(service.services || [])}})">
                    <td><div class="truncate" title="${{escapeHtml(service.name)}}">${{escapeHtml(service.name)}}</div></td>
                    <td><div class="truncate" title="${{escapeHtml(service.datasite)}}">${{escapeHtml(service.datasite)}}</div></td>
                    <td>${{types}}</td>
                    <td>${{pricing}}</td>
                    <td>${{status}}</td>
                    <td>${{tagsDisplay}}</td>
                    <td><div class="truncate" title="${{escapeHtml(service.summary || service.description || '')}}">${{escapeHtml(service.summary || service.description || '')}}</div></td>
                    <td>
                        <button class="copy-btn" onclick="event.stopPropagation(); window['copyServiceName_{container_id}']('${{serviceId}}', this)">Name</button>
                        <button class="copy-btn" onclick="event.stopPropagation(); window['copyServiceExample_{container_id}']('${{serviceId}}', this)">Example</button>
                    </td>
                </tr>`;
            }}).join('');
            
            tbody.innerHTML = rows;
        }}
        
        updatePagination();
    }}

    // Filter functions
    function applyFilters() {{
        const search = document.getElementById('{container_id}-search').value.toLowerCase();
        const serviceType = document.getElementById('{container_id}-service-type').value;
        const pricing = document.getElementById('{container_id}-pricing').value;
        const availability = document.getElementById('{container_id}-availability').value;
        
        filteredServices = allServices.filter(service => {{
            // Search filter
            if (search && !service.name.toLowerCase().includes(search) && 
                !service.datasite.toLowerCase().includes(search) &&
                !service.summary?.toLowerCase().includes(search) &&
                !(service.tags || []).some(tag => tag.toLowerCase().includes(search))) {{
                return false;
            }}
            
            // Service type filter
            if (serviceType && !service.services?.some(s => s.type === serviceType && s.enabled)) {{
                return false;
            }}
            
            // Pricing filter
            if (pricing === 'free' && service.min_pricing > 0) return false;
            if (pricing === 'paid' && service.min_pricing === 0) return false;
            
            // Availability filter
            if (availability && service.health_status !== availability) return false;
            
            return true;
        }});
        
        currentPage = 1;
        renderTable();
    }}

    // Quick filter functions - attach to window for onclick handlers
    window['quickFilter_{container_id}'] = function(type) {{
        clearFilters();
        if (type === 'free') {{
            document.getElementById('{container_id}-pricing').value = 'free';
        }} else if (type === 'online') {{
            document.getElementById('{container_id}-availability').value = 'online';
        }} else if (type === 'chat' || type === 'search') {{
            document.getElementById('{container_id}-service-type').value = type;
        }}
        applyFilters();
    }}

    window['clearFilters_{container_id}'] = function() {{
        document.getElementById('{container_id}-search').value = '';
        document.getElementById('{container_id}-service-type').value = '';
        document.getElementById('{container_id}-pricing').value = '';
        document.getElementById('{container_id}-availability').value = '';
        applyFilters();
    }}

    // Pagination - attach to window for onclick handlers
    window['previousPage_{container_id}'] = function() {{
        if (currentPage > 1) {{
            currentPage--;
            renderTable();
        }}
    }};

    window['nextPage_{container_id}'] = function() {{
        const totalPages = Math.ceil(filteredServices.length / itemsPerPage);
        if (currentPage < totalPages) {{
            currentPage++;
            renderTable();
        }}
    }};

    // Update pagination
    function updatePagination() {{
        const totalPages = Math.ceil(filteredServices.length / itemsPerPage);
        document.getElementById('{container_id}-prev-btn').disabled = currentPage <= 1;
        document.getElementById('{container_id}-next-btn').disabled = currentPage >= totalPages;
        document.getElementById('{container_id}-page-info').textContent = `Page ${{currentPage}} of ${{totalPages || 1}}`;
    }}

    // Quick filter functions - attach to window for inline onclick handlers
    window['quickFilter_{container_id}'] = function(type) {{
        // Reset to first page
        currentPage = 1;
        
        // Clear existing filters first
        document.getElementById('{container_id}-search').value = '';
        document.getElementById('{container_id}-service-type').value = 'all';
        document.getElementById('{container_id}-pricing').value = 'all';
        document.getElementById('{container_id}-availability').value = 'all';
        
        // Apply new filter based on type
        switch(type) {{
            case 'free':
                document.getElementById('{container_id}-pricing').value = 'free';
                break;
            case 'online':
                document.getElementById('{container_id}-availability').value = 'online';
                break;
            case 'chat':
                document.getElementById('{container_id}-service-type').value = 'chat';
                break;
            case 'search':
                document.getElementById('{container_id}-service-type').value = 'search';
                break;
        }}
        
        applyFilters();
    }}
    
    window['clearFilters_{container_id}'] = function() {{
        currentPage = 1;
        document.getElementById('{container_id}-search').value = '';
        document.getElementById('{container_id}-service-type').value = 'all';
        document.getElementById('{container_id}-pricing').value = 'all';
        document.getElementById('{container_id}-availability').value = 'all';
        applyFilters();
    }}

    // Copy functions - attach to window for inline onclick handlers
    window['copyServiceName_{container_id}'] = function(serviceId, button) {{
        if (navigator.clipboard) {{
            navigator.clipboard.writeText(serviceId).then(() => {{
                const original = button.textContent;
                button.textContent = 'Copied!';
                button.classList.add('copied');
                setTimeout(() => {{
                    button.textContent = original;
                    button.classList.remove('copied');
                }}, 1500);
            }});
        }}
    }}

    window['copyServiceExample_{container_id}'] = function(serviceId, button) {{
        // Simple copy of how to call show_example()
        const exampleCode = `service = client.load_service("${{serviceId}}")
service.show_example()`;
        
        if (navigator.clipboard) {{
            navigator.clipboard.writeText(exampleCode).then(() => {{
                const original = button.textContent;
                button.textContent = 'Copied!';
                button.classList.add('copied');
                setTimeout(() => {{
                    button.textContent = original;
                    button.classList.remove('copied');
                }}, 1500);
            }});
        }}
    }}

    // Usage modal - attach to window for inline onclick handlers
    window['showUsageModal_{container_id}'] = function(name, datasite, services) {{
        const serviceId = datasite + '/' + name;
        const hasChat = services.some(s => s.type === 'chat' && s.enabled);
        const hasSearch = services.some(s => s.type === 'search' && s.enabled);
        
        let examples = [];
        if (hasChat) {{
            examples.push(`# Chat with service\\nresponse = client.chat("${{serviceId}}", "Your message here")`);
        }}
        if (hasSearch) {{
            examples.push(`# Search with service\\nresults = client.search("${{serviceId}}", "search query")`);
        }}
        if (examples.length === 0) {{
            examples.push(`# Load service\\nservice = client.load_service("${{serviceId}}")\\nservice.show_example()`);
        }}
        
        alert(`Service: ${{name}}\\nDatasite: ${{datasite}}\\n\\nUsage Examples:\\n${{examples.join('\\n\\n')}}`);
    }}

    // Utility functions
    function escapeHtml(text) {{
        const div = document.createElement('div');
        div.textContent = text || '';
        return div.innerHTML;
    }}

    // Event listeners
    document.getElementById('{container_id}-search').addEventListener('input', applyFilters);
    document.getElementById('{container_id}-service-type').addEventListener('change', applyFilters);
    document.getElementById('{container_id}-pricing').addEventListener('change', applyFilters);
    document.getElementById('{container_id}-availability').addEventListener('change', applyFilters);

    // Initialize and render when DOM is ready
    function initialize() {{
        // Force initial render
        renderTable();
        updatePagination();
    }}
    
    // Call initialize immediately since we're at the end of the body
    initialize();
    
    }})();  // End IIFE
    </script>
    </div>
</body>
</html>
"""