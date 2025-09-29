"""
Theme detection and styling utilities for Jupyter notebooks and Google Colab.

This module provides functionality to detect the current theme in Jupyter/Colab
environments and generate appropriate CSS styles for HTML representations.
"""

def detect_theme() -> str:
    """
    Detect the current theme in Jupyter notebook, Google Colab, or VS Code/Cursor.
    
    Returns:
        str: 'dark' or 'light' theme identifier
    """
    # First check environment variables for manual override
    import os
    theme_override = os.environ.get('SYFT_HUB_THEME', '').lower()
    if theme_override in ['dark', 'light']:
        return theme_override
    try:
        # Try to detect if we're in a Jupyter environment
        from IPython.display import HTML, display
        from IPython import get_ipython
        
        # Get the IPython instance
        ipython = get_ipython()
        if ipython is None:
            return 'light'  # Default fallback
        
        # JavaScript code to detect theme and send it back to Python
        js_code = '''
        <script>
        (function() {
            function detectTheme() {
                // Method 1: Check for VS Code/Cursor dark theme
                const vscodeTheme = document.querySelector('body[data-vscode-theme-kind="vscode-dark"]') ||
                                  document.querySelector('body[data-vscode-theme-kind*="dark"]') ||
                                  document.querySelector('body.vscode-dark') ||
                                  document.querySelector('body.vs-dark') ||
                                  document.querySelector('html[data-theme="dark"]') ||
                                  document.querySelector('html.dark') ||
                                  document.querySelector('body.dark');
                
                if (vscodeTheme) {
                    return 'dark';
                }
                
                // Method 2: Check for Colab dark theme
                const colabDarkTheme = document.querySelector('body[theme="dark"]') || 
                                     document.querySelector('.theme-dark') ||
                                     document.querySelector('[data-theme="dark"]') ||
                                     document.querySelector('body.dark-theme');
                
                if (colabDarkTheme) {
                    return 'dark';
                }
                
                // Method 3: Check for Jupyter lab dark theme
                const jupyterDark = document.querySelector('body[data-jp-theme-name*="dark"]') ||
                                  document.querySelector('.jp-mod-dark') ||
                                  document.querySelector('body.theme-dark');
                
                if (jupyterDark) {
                    return 'dark';
                }
                
                // Method 4: Check computed styles of common elements
                const body = document.body;
                const computedStyle = window.getComputedStyle(body);
                const bgColor = computedStyle.backgroundColor;
                
                // Method 4: Parse background color to determine if dark
                if (bgColor && bgColor !== 'rgba(0, 0, 0, 0)' && bgColor !== 'transparent') {
                    // Parse RGB values
                    const rgb = bgColor.match(/\\d+/g);
                    if (rgb && rgb.length >= 3) {
                        const r = parseInt(rgb[0]);
                        const g = parseInt(rgb[1]);
                        const b = parseInt(rgb[2]);
                        const brightness = (r * 299 + g * 587 + b * 114) / 1000;
                        return brightness < 128 ? 'dark' : 'light';
                    }
                }
                
                // Method 5: Check CSS variables commonly used by themes
                const rootStyles = window.getComputedStyle(document.documentElement);
                const primaryBg = rootStyles.getPropertyValue('--jp-layout-color0') ||
                                rootStyles.getPropertyValue('--colab-primary-surface-color') ||
                                rootStyles.getPropertyValue('--primary-bg-color');
                
                if (primaryBg) {
                    // Similar brightness check for CSS variables
                    const tempDiv = document.createElement('div');
                    tempDiv.style.backgroundColor = primaryBg;
                    document.body.appendChild(tempDiv);
                    const computedBg = window.getComputedStyle(tempDiv).backgroundColor;
                    document.body.removeChild(tempDiv);
                    
                    const rgb = computedBg.match(/\\d+/g);
                    if (rgb && rgb.length >= 3) {
                        const r = parseInt(rgb[0]);
                        const g = parseInt(rgb[1]);
                        const b = parseInt(rgb[2]);
                        const brightness = (r * 299 + g * 587 + b * 114) / 1000;
                        return brightness < 128 ? 'dark' : 'light';
                    }
                }
                
                // Default to light theme
                return 'light';
            }
            
            const theme = detectTheme();
            
            // Store the theme in a global variable that Python can access
            window.syftHubDetectedTheme = theme;
            
            // Also try to communicate with Python through Jupyter's comm system
            if (window.Jupyter && window.Jupyter.notebook && window.Jupyter.notebook.kernel) {
                window.Jupyter.notebook.kernel.execute(
                    `_syft_hub_detected_theme = "${theme}"`
                );
            }
            
            // For Colab, try IPython magic
            if (window.google && window.google.colab) {
                window.google.colab.kernel.invokeFunction('_set_syft_theme', [theme], {});
            }
        })();
        </script>
        '''
        
        # Execute the JavaScript
        display(HTML(js_code))
        
        # Try to get the detected theme from the global variable
        # This might not work immediately due to async nature of JS
        try:
            # Check if the theme was set by JavaScript
            detected_theme = ipython.user_ns.get('_syft_hub_detected_theme', None)
            if detected_theme and detected_theme in ['dark', 'light']:
                return detected_theme
            # If JavaScript didn't detect properly, fall through to environment detection
        except:
            pass
            
    except ImportError:
        pass
    
    # Fallback: try to detect based on environment variables or other indicators
    try:
        import os
        
        # Note: We used to assume VS Code/Cursor uses dark theme, but this was incorrect
        # Now we only return dark if we can actually detect it through JavaScript or env vars
            
        # Some environments might set theme-related environment variables
        if os.environ.get('JUPYTER_THEME', '').lower() in ['dark', 'night']:
            return 'dark'
        if os.environ.get('COLAB_THEME', '').lower() == 'dark':
            return 'dark'
    except:
        pass
    
    # Final fallback to light theme
    return 'light'


def set_theme(theme: str) -> None:
    """
    Manually set the theme for SyftHub widgets.
    
    This is useful in environments where automatic detection doesn't work well,
    such as VS Code/Cursor notebooks. The theme setting persists for the session.
    
    Args:
        theme: 'dark' or 'light'
        
    Examples:
        >>> from syft_hub.utils.theme import set_theme
        >>> set_theme('dark')  # Force dark theme
        >>> set_theme('light') # Force light theme
        
        # For VS Code/Cursor users who want consistent dark theme:
        >>> set_theme('dark')
        >>> client.show_services()  # Will use dark theme
    """
    import os
    if theme.lower() in ['dark', 'light']:
        os.environ['SYFT_HUB_THEME'] = theme.lower()
        print(f"✅ SyftHub theme set to: {theme}")
        print(f"💡 This setting will persist for your current session")
        print(f"💡 Future calls to show_services() will use {theme} theme")
    else:
        raise ValueError("Theme must be 'dark' or 'light'")


def auto_detect_and_set_theme() -> str:
    """
    Automatically detect the current environment theme and set it for SyftHub.
    
    This is useful for ensuring consistent theming across all widgets.
    
    Returns:
        str: The detected theme ('dark' or 'light')
        
    Examples:
        >>> from syft_hub.utils.theme import auto_detect_and_set_theme
        >>> detected_theme = auto_detect_and_set_theme()
        >>> print(f"Auto-detected theme: {detected_theme}")
    """
    detected = detect_theme()
    set_theme(detected)
    return detected


def get_current_theme() -> str:
    """
    Get the currently active theme.
    
    Returns:
        str: 'dark' or 'light'
    """
    return detect_theme()


def get_theme_styles(theme: str = None) -> dict:
    """
    Get CSS styles for light and dark themes.
    
    Args:
        theme: Optional theme override ('light' or 'dark')
        
    Returns:
        dict: CSS styles for both themes
    """
    if theme is None:
        theme = detect_theme()
    
    # Base styles that work for both themes
    base_styles = {
        'font_family': 'system-ui, -apple-system, sans-serif',
        'line_height': '1.5',
        'border_radius': '6px',
        'transition': 'all 0.2s ease-in-out'
    }
    
    # Light theme styles (existing)
    light_theme = {
        'primary_bg': '#fafafa',
        'secondary_bg': '#f8f9fa',
        'border_color': '#e0e0e0',
        'text_primary': '#333',
        'text_secondary': '#666',
        'text_muted': '#6c757d',
        'text_code': '#495057',
        'code_bg': '#f5f5f5',
        'badge_success_bg': '#d4edda',
        'badge_success_text': '#155724',
        'badge_danger_bg': '#f8d7da',
        'badge_danger_text': '#721c24',
        'badge_warning_bg': '#fff3cd',
        'badge_warning_text': '#856404',
        'badge_info_bg': '#e2e3e5',
        'badge_info_text': '#6c757d',
        'warning_bg': '#fef9e7',
        'warning_border': '#e67e22',
        'warning_text': '#e67e22'
    }
    
    # Dark theme styles (improved for better readability)
    dark_theme = {
        'primary_bg': '#2b2b2b',
        'secondary_bg': '#363636',
        'border_color': '#4a4a4a',
        'text_primary': '#f0f0f0',        # Brighter white for better readability
        'text_secondary': '#c0c0c0',      # Brighter secondary text
        'text_muted': '#999',             # Better contrast for muted text
        'text_code': '#f5f5f5',          # Very bright for code readability
        'code_bg': '#1e1e1e',            # Darker code background for better contrast
        'badge_success_bg': '#0d4f14',
        'badge_success_text': '#7bc97f',  # Softer green
        'badge_danger_bg': '#5a1b20',
        'badge_danger_text': '#ff7979',  # Softer red
        'badge_warning_bg': '#4a3c00',
        'badge_warning_text': '#ffa726',  # Better orange contrast
        'badge_info_bg': '#2a2a2a',
        'badge_info_text': '#b0b0b0',
        'warning_bg': '#2a2200',         # Better warning background
        'warning_border': '#ffa726',     # Better orange border
        'warning_text': '#ffc947'        # Better warning text contrast
    }
    
    return {
        'base': base_styles,
        'light': light_theme,
        'dark': dark_theme,
        'current': dark_theme if theme == 'dark' else light_theme
    }


def generate_adaptive_css(class_prefix: str, additional_styles: dict = None) -> str:
    """
    Generate adaptive CSS that automatically switches between light and dark themes.
    
    Uses CSS custom properties to inherit from VS Code/Cursor/Jupyter environments,
    similar to how ipywidgets and other popular libraries handle theming.
    
    Args:
        class_prefix: CSS class prefix for the widget
        additional_styles: Additional CSS rules to include
        
    Returns:
        str: Complete CSS string with both light and dark theme support
    """
    styles = get_theme_styles()
    base = styles['base']
    light = styles['light']
    dark = styles['dark']
    
    # CSS custom properties that inherit from environment (inspired by ipywidgets approach)
    css_variables = '''
    :root {
        /* VS Code/Cursor variables */
        --syft-text-color: var(--vscode-editor-foreground, var(--jp-content-font-color1, #000000));
        --syft-bg-color: var(--vscode-editor-background, var(--jp-layout-color0, #ffffff));
        --syft-border-color: var(--vscode-panel-border, var(--jp-border-color1, #e0e0e0));
        --syft-accent-color: var(--vscode-focusBorder, var(--jp-brand-color1, #007acc));
        
        /* Jupyter variables */
        --syft-notebook-bg: var(--jp-layout-color0, var(--vscode-editor-background, #ffffff));
        --syft-cell-bg: var(--jp-cell-editor-background, var(--vscode-editor-background, #ffffff));
        
        /* Fallback theme detection */
        --syft-is-dark: 0; /* Will be updated by JavaScript */
    }
    
    /* Dark theme detection for environments that don't use CSS variables */
    @media (prefers-color-scheme: dark) {
        :root {
            --syft-is-dark: 1;
        }
    }
    
    /* VS Code dark theme detection */
    body[data-vscode-theme-kind*="dark"]:root,
    body.vscode-dark:root,
    html[data-theme="dark"]:root {
        --syft-is-dark: 1;
        --syft-text-color: var(--vscode-editor-foreground, #cccccc);
        --syft-bg-color: var(--vscode-editor-background, #1e1e1e);
        --syft-border-color: var(--vscode-panel-border, #3c3c3c);
    }
    
    /* JupyterLab dark theme detection */
    body[data-jp-theme-name*="dark"]:root,
    .jp-mod-dark:root {
        --syft-is-dark: 1;
        --syft-text-color: var(--jp-content-font-color1, #cccccc);
        --syft-bg-color: var(--jp-layout-color0, #111111);
        --syft-border-color: var(--jp-border-color1, #3c3c3c);
    }
    
    /* Google Colab dark theme detection */
    body[theme="dark"]:root,
    .theme-dark:root {
        --syft-is-dark: 1;
        --syft-text-color: #e8eaed;
        --syft-bg-color: #202124;
        --syft-border-color: #5f6368;
    }
    '''
    
    # JavaScript for theme detection
    theme_detection_js = '''
    <script>
    (function() {
        function detectAndApplyTheme() {
            // Check for dark theme using multiple methods
            let isDark = false;
            
            // Method 1: Check body/html attributes
            isDark = isDark || document.querySelector('body[theme="dark"]') !== null;
            isDark = isDark || document.querySelector('.theme-dark') !== null;
            isDark = isDark || document.querySelector('[data-theme="dark"]') !== null;
            isDark = isDark || document.querySelector('body.dark-theme') !== null;
            isDark = isDark || document.querySelector('body[data-jp-theme-name*="dark"]') !== null;
            isDark = isDark || document.querySelector('.jp-mod-dark') !== null;
            isDark = isDark || document.querySelector('body[data-vscode-theme-kind*="dark"]') !== null;
            isDark = isDark || document.querySelector('body.vscode-dark') !== null;
            
            // Method 2: Check CSS media query (prefers-color-scheme)
            if (!isDark && window.matchMedia) {
                isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            }
            
            // Method 3: For Colab - check computed background color of body
            if (!isDark) {
                try {
                    const bodyBg = window.getComputedStyle(document.body).backgroundColor;
                    if (bodyBg && bodyBg !== 'rgba(0, 0, 0, 0)' && bodyBg !== 'transparent') {
                        const rgb = bodyBg.match(/\\d+/g);
                        if (rgb && rgb.length >= 3) {
                            const r = parseInt(rgb[0]);
                            const g = parseInt(rgb[1]);
                            const b = parseInt(rgb[2]);
                            const brightness = (r * 299 + g * 587 + b * 114) / 1000;
                            isDark = brightness < 128;
                        }
                    }
                } catch (e) {
                    // Ignore errors
                }
            }
            
            // Apply theme to all widgets
            const widgets = document.querySelectorAll('.syft-widget');
            widgets.forEach(widget => {
                widget.setAttribute('data-theme', isDark ? 'dark' : 'light');
            });
            
            // Also set a CSS class for easier styling
            widgets.forEach(widget => {
                if (isDark) {
                    widget.classList.add('syft-dark-mode');
                } else {
                    widget.classList.remove('syft-dark-mode');
                }
            });
        }
        
        // Apply theme immediately
        detectAndApplyTheme();
        
        // Apply again after a short delay (for Colab compatibility)
        setTimeout(detectAndApplyTheme, 100);
        setTimeout(detectAndApplyTheme, 500);
        
        // Apply when DOM is fully loaded
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', detectAndApplyTheme);
        }
        
        // Watch for theme changes
        try {
            const observer = new MutationObserver(detectAndApplyTheme);
            observer.observe(document.body, {
                attributes: true,
                attributeFilter: ['theme', 'data-jp-theme-name', 'class', 'data-theme', 'data-vscode-theme-kind']
            });
            observer.observe(document.documentElement, {
                attributes: true,
                attributeFilter: ['theme', 'data-jp-theme-name', 'class', 'data-theme']
            });
        } catch (e) {
            // If MutationObserver fails, fall back to periodic checking
            setInterval(detectAndApplyTheme, 1000);
        }
        
        // Listen for system theme changes
        if (window.matchMedia) {
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', detectAndApplyTheme);
        }
    })();
    </script>
    '''
    
    # Generate CSS with CSS custom properties (modern approach)
    css = f'''
    <style>
        {css_variables}
        
        .syft-widget {{
            font-family: var(--syft-font-family, {base['font_family']});
            line-height: {base['line_height']};
            transition: {base['transition']};
        }}
        
        /* Default theme using CSS custom properties */
        .{class_prefix}-widget {{
            color: var(--syft-text-color, {light['text_primary']});
            background: var(--syft-bg-color, {light['primary_bg']});
            border: 1px solid var(--syft-border-color, {light['border_color']});
            border-radius: {base['border_radius']};
            padding: 16px;
            margin: 8px 0;
        }}
        
        /* Ensure all child elements have consistent left alignment */
        .{class_prefix}-widget > * {{
            margin-left: 0;
            padding-left: 0;
        }}
        
        .widget-title {{
            font-size: 14px;
            font-weight: 600;
            margin: 0 0 12px 0;
            padding: 0;
            color: var(--syft-text-color, {light['text_primary']});
        }}
        .status-line {{
            display: flex;
            align-items: flex-start;
            margin: 6px 0;
            padding: 0;
            font-size: 11px;
        }}
        .status-label {{
            color: var(--syft-text-color, {light['text_secondary']});
            opacity: 0.8;
            min-width: 140px;
            margin-right: 12px;
            padding: 0;
            flex-shrink: 0;
        }}
        .status-value {{
            font-family: monospace;
            color: var(--syft-text-color, {light['text_primary']});
            word-break: break-word;
            margin: 0;
            padding: 0;
        }}
        
        /* Ensure content-preview elements align with status-line elements */
        .{class_prefix}-widget .content-preview {{
            background: {light['secondary_bg']};
            border: 1px solid {light['border_color']};
            border-radius: 4px;
            padding: 8px;
            margin: 6px 0;
            margin-left: 0;
            font-family: inherit;
            font-size: 11px;
            color: {light['text_code']};
            white-space: pre-wrap;
        }}
        .status-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 11px;
            margin-left: 8px;
        }}
        .badge-ready {{
            background: {light['badge_success_bg']};
            color: {light['badge_success_text']};
        }}
        .badge-not-ready {{
            background: {light['badge_danger_bg']};
            color: {light['badge_danger_text']};
        }}
        .badge-online {{
            background: {light['badge_success_bg']};
            color: {light['badge_success_text']};
        }}
        .badge-offline {{
            background: {light['badge_danger_bg']};
            color: {light['badge_danger_text']};
        }}
        .badge-timeout {{
            background: {light['badge_warning_bg']};
            color: {light['badge_warning_text']};
        }}
        .badge-unknown {{
            background: {light['badge_info_bg']};
            color: {light['badge_info_text']};
        }}
        .docs-section {{
            margin-top: 16px;
            padding-top: 12px;
            border-top: 1px solid {light['border_color']};
            font-size: 11px;
            color: {light['text_secondary']};
        }}
        .command-code {{
            font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
            color: {light['text_primary']};
            background: transparent !important;
            font-size: 12px;
        }}
        .comment {{
            color: {light['text_muted']};
            font-style: italic;
            font-size: 10px;
        }}
        
        /* Markdown content styling */
        .markdown-content {{
            font-size: 12px;
            line-height: 1.4;
        }}
        .markdown-content p {{
            margin: 6px 0;
        }}
        .markdown-content ul, .markdown-content ol {{
            margin: 6px 0;
            padding-left: 20px;
        }}
        .markdown-content li {{
            margin: 2px 0;
        }}
        .markdown-content h1, .markdown-content h2, .markdown-content h3, 
        .markdown-content h4, .markdown-content h5, .markdown-content h6 {{
            margin: 8px 0 4px 0;
            font-weight: 600;
        }}
        .markdown-content code {{
            font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
            font-size: 11px;
            background: rgba(0, 0, 0, 0.05);
            padding: 1px 3px;
            border-radius: 2px;
        }}
        .markdown-content pre {{
            background: rgba(0, 0, 0, 0.05);
            padding: 8px;
            border-radius: 4px;
            margin: 8px 0;
            overflow-x: auto;
        }}
        .markdown-content pre code {{
            background: transparent;
            padding: 0;
        }}
        .code-block {{
            position: relative;
            margin: 12px 0;
            padding: 16px 36px 16px 16px;
            background: transparent;
            border-radius: 6px;
            line-height: 1.5;
            font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
            font-size: 12px;
            overflow-x: auto;
        }}
        .code-block .command-code {{
            background: transparent !important;
            font-family: inherit;
            font-size: inherit;
            line-height: inherit;
            color: inherit;
            display: block;
            white-space: pre-wrap;
            word-break: break-word;
        }}
        .copy-btn {{
            position: absolute;
            top: 4px;
            right: 4px;
            background: rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(0, 0, 0, 0.2);
            border-radius: 2px;
            padding: 2px;
            font-size: 12px;
            cursor: pointer;
            opacity: 0.6;
            transition: opacity 0.2s ease;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            width: 20px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10;
        }}
        .copy-btn:hover {{
            opacity: 1;
            background: rgba(0, 0, 0, 0.15);
        }}
        
        /* Dark theme - primary selectors for Colab compatibility */
        .syft-widget.syft-dark-mode .{class_prefix}-widget {{
            color: {dark['text_primary']} !important;
            background: {dark['primary_bg']} !important;
            border-color: {dark['border_color']} !important;
        }}
        .syft-widget.syft-dark-mode .widget-title {{
            color: {dark['text_primary']} !important;
        }}
        .syft-widget.syft-dark-mode .status-label {{
            color: {dark['text_secondary']} !important;
        }}
        .syft-widget.syft-dark-mode .status-value {{
            color: #ffffff !important;
            font-weight: 500 !important;
        }}
        .syft-widget.syft-dark-mode .markdown-content {{
            color: {dark['text_primary']} !important;
        }}
        .syft-widget.syft-dark-mode .command-code {{
            color: {dark['text_primary']} !important;
        }}
        .syft-widget.syft-dark-mode .badge-ready,
        .syft-widget.syft-dark-mode .badge-complete {{
            background: {dark['badge_success_bg']} !important;
            color: {dark['badge_success_text']} !important;
        }}
        .syft-widget.syft-dark-mode .badge-not-ready {{
            background: {dark['badge_danger_bg']} !important;
            color: {dark['badge_danger_text']} !important;
        }}
        .syft-widget.syft-dark-mode .badge-online {{
            background: {dark['badge_success_bg']} !important;
            color: {dark['badge_success_text']} !important;
        }}
        .syft-widget.syft-dark-mode .badge-offline {{
            background: {dark['badge_danger_bg']} !important;
            color: {dark['badge_danger_text']} !important;
        }}
        
        /* Dark theme - fallback selectors for other environments */
        .syft-widget[data-theme="dark"] .{class_prefix}-widget,
        .syft-widget.syft-dark-mode .{class_prefix}-widget,
        .theme-dark .{class_prefix}-widget,
        body[theme="dark"] .{class_prefix}-widget,
        body[data-jp-theme-name*="dark"] .{class_prefix}-widget,
        body[data-vscode-theme-kind*="dark"] .{class_prefix}-widget,
        body.vscode-dark .{class_prefix}-widget,
        body.vs-dark .{class_prefix}-widget,
        html[data-theme="dark"] .{class_prefix}-widget,
        html.dark .{class_prefix}-widget,
        body.dark .{class_prefix}-widget,
        .jp-mod-dark .{class_prefix}-widget {{
            color: {dark['text_primary']};
            background: {dark['primary_bg']};
            border-color: {dark['border_color']};
        }}
        .syft-widget[data-theme="dark"] .widget-title,
        .syft-widget.syft-dark-mode .widget-title,
        .theme-dark .widget-title,
        body[theme="dark"] .widget-title,
        body[data-jp-theme-name*="dark"] .widget-title,
        body[data-vscode-theme-kind*="dark"] .widget-title,
        body.vscode-dark .widget-title,
        body.vs-dark .widget-title,
        html[data-theme="dark"] .widget-title,
        html.dark .widget-title,
        body.dark .widget-title,
        .jp-mod-dark .widget-title {{
            color: {dark['text_primary']};
        }}
        .syft-widget[data-theme="dark"] .status-label,
        .syft-widget.syft-dark-mode .status-label,
        .theme-dark .status-label,
        body[theme="dark"] .status-label,
        body[data-jp-theme-name*="dark"] .status-label,
        .jp-mod-dark .status-label {{
            color: {dark['text_secondary']};
        }}
        .syft-widget[data-theme="dark"] .status-value,
        .syft-widget.syft-dark-mode .status-value,
        .theme-dark .status-value,
        body[theme="dark"] .status-value,
        body[data-jp-theme-name*="dark"] .status-value,
        body[data-vscode-theme-kind*="dark"] .status-value,
        body.vscode-dark .status-value,
        body.vs-dark .status-value,
        html[data-theme="dark"] .status-value,
        html.dark .status-value,
        body.dark .status-value,
        .jp-mod-dark .status-value {{
            color: #ffffff !important;
            font-weight: 500 !important;
        }}
        .syft-widget[data-theme="dark"] .badge-ready,
        .syft-widget[data-theme="dark"] .badge-complete,
        .theme-dark .badge-ready,
        .theme-dark .badge-complete,
        body[theme="dark"] .badge-ready,
        body[theme="dark"] .badge-complete,
        body[data-jp-theme-name*="dark"] .badge-ready,
        body[data-jp-theme-name*="dark"] .badge-complete,
        .jp-mod-dark .badge-ready,
        .jp-mod-dark .badge-complete {{
            background: {dark['badge_success_bg']} !important;
            color: {dark['badge_success_text']} !important;
        }}
        .syft-widget[data-theme="dark"] .badge-not-ready,
        .theme-dark .badge-not-ready,
        body[theme="dark"] .badge-not-ready,
        body[data-jp-theme-name*="dark"] .badge-not-ready,
        .jp-mod-dark .badge-not-ready {{
            background: {dark['badge_danger_bg']};
            color: {dark['badge_danger_text']};
        }}
        .syft-widget[data-theme="dark"] .badge-online,
        .theme-dark .badge-online,
        body[theme="dark"] .badge-online,
        body[data-jp-theme-name*="dark"] .badge-online,
        .jp-mod-dark .badge-online {{
            background: {dark['badge_success_bg']};
            color: {dark['badge_success_text']};
        }}
        .syft-widget[data-theme="dark"] .badge-offline,
        .theme-dark .badge-offline,
        body[theme="dark"] .badge-offline,
        body[data-jp-theme-name*="dark"] .badge-offline,
        .jp-mod-dark .badge-offline {{
            background: {dark['badge_danger_bg']};
            color: {dark['badge_danger_text']};
        }}
        .syft-widget[data-theme="dark"] .badge-timeout,
        .theme-dark .badge-timeout,
        body[theme="dark"] .badge-timeout,
        body[data-jp-theme-name*="dark"] .badge-timeout,
        .jp-mod-dark .badge-timeout {{
            background: {dark['badge_warning_bg']};
            color: {dark['badge_warning_text']};
        }}
        .syft-widget[data-theme="dark"] .badge-unknown,
        .theme-dark .badge-unknown,
        body[theme="dark"] .badge-unknown,
        body[data-jp-theme-name*="dark"] .badge-unknown,
        .jp-mod-dark .badge-unknown {{
            background: {dark['badge_info_bg']};
            color: {dark['badge_info_text']};
        }}
        .syft-widget[data-theme="dark"] .docs-section,
        .theme-dark .docs-section,
        body[theme="dark"] .docs-section,
        body[data-jp-theme-name*="dark"] .docs-section,
        .jp-mod-dark .docs-section {{
            border-top-color: {dark['border_color']};
            color: {dark['text_secondary']};
        }}
        .syft-widget[data-theme="dark"] .command-code,
        .theme-dark .command-code,
        body[theme="dark"] .command-code,
        body[data-jp-theme-name*="dark"] .command-code,
        body[data-vscode-theme-kind*="dark"] .command-code,
        body.vscode-dark .command-code,
        body.vs-dark .command-code,
        html[data-theme="dark"] .command-code,
        html.dark .command-code,
        body.dark .command-code,
        .jp-mod-dark .command-code {{
            background: transparent !important;
            color: {dark['text_primary']};
            font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
            font-size: 12px;
        }}
        .syft-widget[data-theme="dark"] .comment,
        .theme-dark .comment,
        body[theme="dark"] .comment,
        body[data-jp-theme-name*="dark"] .comment,
        .jp-mod-dark .comment {{
            color: {dark['text_muted']};
            font-style: italic;
            font-size: 10px;
        }}
        
        /* Dark mode markdown content styling */
        .syft-widget[data-theme="dark"] .markdown-content,
        .theme-dark .markdown-content,
        body[theme="dark"] .markdown-content,
        body[data-jp-theme-name*="dark"] .markdown-content,
        body[data-vscode-theme-kind*="dark"] .markdown-content,
        body.vscode-dark .markdown-content,
        body.vs-dark .markdown-content,
        html[data-theme="dark"] .markdown-content,
        html.dark .markdown-content,
        body.dark .markdown-content,
        .jp-mod-dark .markdown-content {{
            color: {dark['text_primary']};
        }}
        .syft-widget[data-theme="dark"] .markdown-content code,
        .theme-dark .markdown-content code,
        body[theme="dark"] .markdown-content code,
        body[data-jp-theme-name*="dark"] .markdown-content code,
        body[data-vscode-theme-kind*="dark"] .markdown-content code,
        body.vscode-dark .markdown-content code,
        body.vs-dark .markdown-content code,
        html[data-theme="dark"] .markdown-content code,
        html.dark .markdown-content code,
        body.dark .markdown-content code,
        .jp-mod-dark .markdown-content code {{
            background: rgba(255, 255, 255, 0.1);
            color: {dark['text_primary']};
        }}
        .syft-widget[data-theme="dark"] .markdown-content pre,
        .theme-dark .markdown-content pre,
        body[theme="dark"] .markdown-content pre,
        body[data-jp-theme-name*="dark"] .markdown-content pre,
        body[data-vscode-theme-kind*="dark"] .markdown-content pre,
        body.vscode-dark .markdown-content pre,
        body.vs-dark .markdown-content pre,
        html[data-theme="dark"] .markdown-content pre,
        html.dark .markdown-content pre,
        body.dark .markdown-content pre,
        .jp-mod-dark .markdown-content pre {{
            background: rgba(255, 255, 255, 0.1);
            color: {dark['text_primary']};
        }}
        .syft-widget[data-theme="dark"] .code-block,
        .theme-dark .code-block,
        body[theme="dark"] .code-block,
        body[data-jp-theme-name*="dark"] .code-block,
        body[data-vscode-theme-kind*="dark"] .code-block,
        body.vscode-dark .code-block,
        body.vs-dark .code-block,
        html[data-theme="dark"] .code-block,
        html.dark .code-block,
        body.dark .code-block,
        .jp-mod-dark .code-block {{
            background: transparent !important;
            color: {dark['text_primary']} !important;
            margin: 12px 0;
            padding: 16px 36px 16px 16px;
            border-radius: 6px;
            line-height: 1.5;
            font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
            font-size: 12px;
            overflow-x: auto;
        }}
        .syft-widget[data-theme="dark"] .code-block .command-code,
        .theme-dark .code-block .command-code,
        body[theme="dark"] .code-block .command-code,
        body[data-jp-theme-name*="dark"] .code-block .command-code,
        body[data-vscode-theme-kind*="dark"] .code-block .command-code,
        body.vscode-dark .code-block .command-code,
        body.vs-dark .code-block .command-code,
        html[data-theme="dark"] .code-block .command-code,
        html.dark .code-block .command-code,
        body.dark .code-block .command-code,
        .jp-mod-dark .code-block .command-code {{
            background: transparent !important;
            color: {dark['text_primary']} !important;
            font-family: inherit;
            font-size: inherit;
            line-height: inherit;
            display: block;
            white-space: pre-wrap;
            word-break: break-word;
        }}
        .syft-widget[data-theme="dark"] .copy-btn,
        .theme-dark .copy-btn,
        body[theme="dark"] .copy-btn,
        body[data-jp-theme-name*="dark"] .copy-btn,
        body[data-vscode-theme-kind*="dark"] .copy-btn,
        body.vscode-dark .copy-btn,
        body.vs-dark .copy-btn,
        html[data-theme="dark"] .copy-btn,
        html.dark .copy-btn,
        body.dark .copy-btn,
        .jp-mod-dark .copy-btn {{
            position: absolute;
            top: 4px;
            right: 4px;
            background: rgba(255, 255, 255, 0.1) !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            color: {dark['text_primary']} !important;
            border-radius: 2px;
            padding: 2px;
            font-size: 12px;
            cursor: pointer;
            opacity: 0.6;
            transition: opacity 0.2s ease;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            width: 20px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10;
        }}
        .syft-widget[data-theme="dark"] .copy-btn:hover,
        .theme-dark .copy-btn:hover,
        body[theme="dark"] .copy-btn:hover,
        body[data-jp-theme-name*="dark"] .copy-btn:hover,
        body[data-vscode-theme-kind*="dark"] .copy-btn:hover,
        body.vscode-dark .copy-btn:hover,
        body.vs-dark .copy-btn:hover,
        html[data-theme="dark"] .copy-btn:hover,
        html.dark .copy-btn:hover,
        body.dark .copy-btn:hover,
        .jp-mod-dark .copy-btn:hover {{
            opacity: 1 !important;
            background: rgba(255, 255, 255, 0.15) !important;
        }}
        /* Dark theme margin consistency */
        .syft-widget[data-theme="dark"] .{class_prefix}-widget > *,
        .theme-dark .{class_prefix}-widget > *,
        body[theme="dark"] .{class_prefix}-widget > *,
        body[data-jp-theme-name*="dark"] .{class_prefix}-widget > *,
        body[data-vscode-theme-kind*="dark"] .{class_prefix}-widget > *,
        body.vscode-dark .{class_prefix}-widget > *,
        body.vs-dark .{class_prefix}-widget > *,
        html[data-theme="dark"] .{class_prefix}-widget > *,
        html.dark .{class_prefix}-widget > *,
        body.dark .{class_prefix}-widget > *,
        .jp-mod-dark .{class_prefix}-widget > * {{
            margin-left: 0 !important;
            padding-left: 0 !important;
        }}
        
        .syft-widget[data-theme="dark"] .content-preview,
        .theme-dark .content-preview,
        body[theme="dark"] .content-preview,
        body[data-jp-theme-name*="dark"] .content-preview,
        body[data-vscode-theme-kind*="dark"] .content-preview,
        body.vscode-dark .content-preview,
        body.vs-dark .content-preview,
        html[data-theme="dark"] .content-preview,
        html.dark .content-preview,
        body.dark .content-preview,
        .jp-mod-dark .content-preview {{
            background: {dark['secondary_bg']};
            border-color: {dark['border_color']};
            color: {dark['text_code']};
            margin-left: 0 !important;
        }}
        
        /* Dark theme overrides for inline styled elements */
        .syft-widget[data-theme="dark"] div[style*="color: #e67e22"],
        .theme-dark div[style*="color: #e67e22"],
        body[theme="dark"] div[style*="color: #e67e22"],
        body[data-jp-theme-name*="dark"] div[style*="color: #e67e22"],
        body[data-vscode-theme-kind*="dark"] div[style*="color: #e67e22"],
        body.vscode-dark div[style*="color: #e67e22"],
        body.vs-dark div[style*="color: #e67e22"],
        html[data-theme="dark"] div[style*="color: #e67e22"],
        html.dark div[style*="color: #e67e22"],
        body.dark div[style*="color: #e67e22"],
        .jp-mod-dark div[style*="color: #e67e22"] {{
            color: {dark['warning_text']} !important;
            background: {dark['warning_bg']} !important;
            border: 1px solid {dark['warning_border']} !important;
        }}
        
        /* Better styling for inline monospace spans (code elements) - less highlighted */
        .syft-widget[data-theme="dark"] span[style*="font-family: monospace"],
        .syft-widget[data-theme="dark"] .command-code,
        .theme-dark span[style*="font-family: monospace"],
        .theme-dark .command-code,
        body[theme="dark"] span[style*="font-family: monospace"],
        body[theme="dark"] .command-code,
        body[data-jp-theme-name*="dark"] span[style*="font-family: monospace"],
        body[data-jp-theme-name*="dark"] .command-code,
        body[data-vscode-theme-kind*="dark"] span[style*="font-family: monospace"],
        body[data-vscode-theme-kind*="dark"] .command-code,
        body.vscode-dark span[style*="font-family: monospace"],
        body.vscode-dark .command-code,
        body.vs-dark span[style*="font-family: monospace"],
        body.vs-dark .command-code,
        html[data-theme="dark"] span[style*="font-family: monospace"],
        html[data-theme="dark"] .command-code,
        html.dark span[style*="font-family: monospace"],
        html.dark .command-code,
        body.dark span[style*="font-family: monospace"],
        body.dark .command-code,
        .jp-mod-dark span[style*="font-family: monospace"],
        .jp-mod-dark .command-code {{
            background: {dark['secondary_bg']} !important;
            color: #b8c5d1 !important;
            padding: 2px 4px !important;
            border-radius: 3px !important;
            border: 1px solid {dark['border_color']} !important;
            font-weight: normal !important;
        }}
        
        /* Fix inline comment colors (gray text) */
        .syft-widget[data-theme="dark"] span[style*="color: #6c757d"],
        .theme-dark span[style*="color: #6c757d"],
        body[theme="dark"] span[style*="color: #6c757d"],
        body[data-jp-theme-name*="dark"] span[style*="color: #6c757d"],
        body[data-vscode-theme-kind*="dark"] span[style*="color: #6c757d"],
        body.vscode-dark span[style*="color: #6c757d"],
        body.vs-dark span[style*="color: #6c757d"],
        html[data-theme="dark"] span[style*="color: #6c757d"],
        html.dark span[style*="color: #6c757d"],
        body.dark span[style*="color: #6c757d"],
        .jp-mod-dark span[style*="color: #6c757d"] {{
            color: {dark['text_muted']} !important;
        }}
        
        /* Fix inline price colors (green/red) */
        .syft-widget[data-theme="dark"] span[style*="color: #28a745"],
        .theme-dark span[style*="color: #28a745"],
        body[theme="dark"] span[style*="color: #28a745"],
        body[data-jp-theme-name*="dark"] span[style*="color: #28a745"],
        body[data-vscode-theme-kind*="dark"] span[style*="color: #28a745"],
        body.vscode-dark span[style*="color: #28a745"],
        body.vs-dark span[style*="color: #28a745"],
        html[data-theme="dark"] span[style*="color: #28a745"],
        html.dark span[style*="color: #28a745"],
        body.dark span[style*="color: #28a745"],
        .jp-mod-dark span[style*="color: #28a745"] {{
            color: {dark['badge_success_text']} !important;
        }}
        
        .syft-widget[data-theme="dark"] span[style*="color: #dc3545"],
        .theme-dark span[style*="color: #dc3545"],
        body[theme="dark"] span[style*="color: #dc3545"],
        body[data-jp-theme-name*="dark"] span[style*="color: #dc3545"],
        body[data-vscode-theme-kind*="dark"] span[style*="color: #dc3545"],
        body.vscode-dark span[style*="color: #dc3545"],
        body.vs-dark span[style*="color: #dc3545"],
        html[data-theme="dark"] span[style*="color: #dc3545"],
        html.dark span[style*="color: #dc3545"],
        body.dark span[style*="color: #dc3545"],
        .jp-mod-dark span[style*="color: #dc3545"] {{
            color: {dark['badge_danger_text']} !important;
        }}
        
        /* Fix inline secondary text colors */
        .syft-widget[data-theme="dark"] div[style*="color: #666"],
        .theme-dark div[style*="color: #666"],
        body[theme="dark"] div[style*="color: #666"],
        body[data-jp-theme-name*="dark"] div[style*="color: #666"],
        body[data-vscode-theme-kind*="dark"] div[style*="color: #666"],
        body.vscode-dark div[style*="color: #666"],
        body.vs-dark div[style*="color: #666"],
        html[data-theme="dark"] div[style*="color: #666"],
        html.dark div[style*="color: #666"],
        body.dark div[style*="color: #666"],
        .jp-mod-dark div[style*="color: #666"] {{
            color: {dark['text_secondary']} !important;
        }}
        
        /* Additional overrides for ANY monospace elements that might be too bright */
        .syft-widget[data-theme="dark"] *[style*="font-family: monospace"],
        .syft-widget[data-theme="dark"] *[style*="font-weight: 600"],
        .theme-dark *[style*="font-family: monospace"],
        .theme-dark *[style*="font-weight: 600"],
        body[theme="dark"] *[style*="font-family: monospace"],
        body[theme="dark"] *[style*="font-weight: 600"],
        body[data-jp-theme-name*="dark"] *[style*="font-family: monospace"],
        body[data-jp-theme-name*="dark"] *[style*="font-weight: 600"],
        body[data-vscode-theme-kind*="dark"] *[style*="font-family: monospace"],
        body[data-vscode-theme-kind*="dark"] *[style*="font-weight: 600"],
        body.vscode-dark *[style*="font-family: monospace"],
        body.vscode-dark *[style*="font-weight: 600"],
        body.vs-dark *[style*="font-family: monospace"],
        body.vs-dark *[style*="font-weight: 600"],
        html[data-theme="dark"] *[style*="font-family: monospace"],
        html[data-theme="dark"] *[style*="font-weight: 600"],
        html.dark *[style*="font-family: monospace"],
        html.dark *[style*="font-weight: 600"],
        body.dark *[style*="font-family: monospace"],
        body.dark *[style*="font-weight: 600"],
        .jp-mod-dark *[style*="font-family: monospace"],
        .jp-mod-dark *[style*="font-weight: 600"] {{
            background: {dark['secondary_bg']} !important;
            color: #a8b3c1 !important;
            font-weight: normal !important;
            border: 1px solid {dark['border_color']} !important;
        }}
        
        /* Ensure attribute values are very readable */
        .syft-widget[data-theme="dark"] .status-value *,
        .theme-dark .status-value *,
        body[theme="dark"] .status-value *,
        body[data-jp-theme-name*="dark"] .status-value *,
        body[data-vscode-theme-kind*="dark"] .status-value *,
        body.vscode-dark .status-value *,
        body.vs-dark .status-value *,
        html[data-theme="dark"] .status-value *,
        html.dark .status-value *,
        body.dark .status-value *,
        .jp-mod-dark .status-value * {{
            color: #ffffff !important;
        }}
    </style>
    {theme_detection_js}
    '''
    
    # Add any additional styles
    if additional_styles:
        css += '\n<style>\n'
        for selector, rules in additional_styles.items():
            css += f'{selector} {{\n'
            for prop, value in rules.items():
                css += f'    {prop}: {value};\n'
            css += '}\n'
        css += '</style>'
    
    return css
