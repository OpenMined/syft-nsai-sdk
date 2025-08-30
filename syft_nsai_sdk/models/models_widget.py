"""
Models widget HTML template for the SyftBox NSAI SDK.
"""

from typing import List, Optional
import json
import random
import uuid


def get_models_widget_html(
    models: Optional[List] = None,
    service_type: Optional[str] = None,
    owner: Optional[str] = None,
    tags: Optional[List[str]] = None,
    max_cost: Optional[float] = None,
    free_only: bool = False,
    health_check: str = "auto",
    include_disabled: bool = False,
    page: int = 1,
    items_per_page: int = 50,
    current_user_email: str = "",
) -> str:
    """Generate the models widget HTML for web serving.
    
    Note: Filtering parameters (service_type, owner, tags, etc.) are optional.
    When not provided, the widget will display all models without additional filtering.
    """
    
    container_id = f"syft_models_{uuid.uuid4().hex[:8]}"

    # Non-obvious tips for users
    tips = [
        'Use quotation marks to search for exact phrases like "machine learning"',
        "Multiple words without quotes searches for models containing ALL words",
        "Press Tab in search boxes for auto-completion suggestions",
        "Tab completion in Owner filter shows all available datasite emails",
        "Click any row to copy its model identifier to clipboard",
        "Health check shows real-time model availability",
        "Free models have $0.00 pricing",
        "Tags help categorize models by purpose or technology",
        "Owner shows who published the model",
        "Services column shows available capabilities (chat, search)",
        "Status shows if model is active and healthy",
        "Use client.chat() or client.search() to interact with models",
        "Filter by max_cost to stay within budget",
        "Set free_only=True to see only free models",
        "Health status: ✅ Online, ❌ Offline, ⏱️ Timeout, ❓ Unknown",
        "Models with multiple services can do both chat and search"
    ]

    # Pick a random tip for footer
    footer_tip = random.choice(tips)
    show_footer_tip = random.random() < 0.5  # 50% chance

    # Handle optional filtering parameters
    # If not provided, set defaults that show all models
    if service_type is None:
        service_type = ""
    if owner is None:
        owner = ""
    if tags is None:
        tags = []
    if max_cost is None:
        max_cost = 0  # 0 means no cost limit
    if health_check is None:
        health_check = "auto"

    # Generate complete HTML with the widget
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SyftBox Models</title>
    <style>
    body {{
        background-color: #ffffff;
        color: #000000;
        margin: 0;
        padding: 0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }}

    @keyframes float {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-8px); }}
    }}

    .syftbox-logo {{
        animation: float 3s ease-in-out infinite;
        filter: drop-shadow(0 4px 12px rgba(0, 0, 0, 0.15));
    }}

    .progress-bar-gradient {{
        background: linear-gradient(90deg, #3b82f6 0%, #10b981 100%);
        transition: width 0.4s ease-out;
        border-radius: 3px;
    }}

    #{container_id} * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}

    #{container_id} {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 12px;
        background: #ffffff;
        color: #000000;
        display: flex;
        flex-direction: column;
        width: 100%;
        height: 100vh;
        margin: 0;
        border: none;
        border-radius: 8px;
        overflow: hidden;
    }}

    #{container_id} .search-controls {{
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        padding: 0.75rem;
        background: #f8f9fa;
        border-bottom: 1px solid #e5e7eb;
        flex-shrink: 0;
    }}

    #{container_id} .search-controls input, #{container_id} .search-controls select {{
        flex: 1;
        min-width: 150px;
        padding: 0.5rem;
        border: 1px solid #d1d5db;
        border-radius: 0.25rem;
        font-size: 0.875rem;
        background: #ffffff;
        color: #000000;
    }}

    #{container_id} .table-container {{
        flex: 1;
        overflow-y: auto;
        overflow-x: auto;
        background: #ffffff;
        min-height: 0;
    }}

    #{container_id} table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.75rem;
        table-layout: fixed;
        min-width: 800px; /* Ensure minimum width for readability */
    }}

    /* Improved column widths - balanced to fill full width */
    #{container_id} th:nth-child(1) {{ width: 15%; }} /* Model Name */
    #{container_id} th:nth-child(2) {{ width: 12%; }} /* Owner */
    #{container_id} th:nth-child(3) {{ width: 12%; }} /* Services */
    #{container_id} th:nth-child(4) {{ width: 10%; }} /* Pricing */
    #{container_id} th:nth-child(5) {{ width: 10%; }} /* Status */
    #{container_id} th:nth-child(6) {{ width: 15%; }} /* Tags */
    #{container_id} th:nth-child(7) {{ width: 26%; }} /* Summary */

    #{container_id} thead {{
        background: #f8f9fa;
        border-bottom: 1px solid #e5e7eb;
    }}

    #{container_id} th {{
        text-align: left;
        padding: 0.375rem 0.25rem;
        font-weight: 500;
        font-size: 0.75rem;
        border-bottom: 1px solid #e5e7eb;
        position: sticky;
        top: 0;
        background: #f8f9fa;
        z-index: 10;
        color: #000000;
    }}

    #{container_id} td {{
        padding: 0.375rem 0.25rem;
        border-bottom: 1px solid #f3f4f6;
        vertical-align: top;
        font-size: 0.75rem;
        text-align: left;
        color: #000000;
    }}

    #{container_id} tbody tr {{
        transition: background-color 0.15s;
        cursor: pointer;
    }}

    #{container_id} tbody tr:hover {{
        background: rgba(0, 0, 0, 0.03);
    }}

    #{container_id} .pagination {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem;
        border-top: 1px solid #e5e7eb;
        background: rgba(0, 0, 0, 0.02);
        flex-shrink: 0;
    }}

    #{container_id} .pagination button {{
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        border: 1px solid #e5e7eb;
        background: white;
        color: #000000;
        cursor: pointer;
        transition: all 0.15s;
    }}

    #{container_id} .pagination button:hover:not(:disabled) {{
        background: #f3f4f6;
    }}

    #{container_id} .pagination button:disabled {{
        opacity: 0.5;
        cursor: not-allowed;
    }}

    #{container_id} .pagination .page-info {{
        font-size: 0.75rem;
        color: #6b7280;
    }}

    #{container_id} .pagination .status {{
        font-size: 0.75rem;
        color: #9ca3af;
        font-style: italic;
        opacity: 0.8;
        text-align: center;
        flex: 1;
    }}

    /* Enhanced truncation with tooltips */
    #{container_id} .truncate {{
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        position: relative;
        cursor: help;
        max-width: 100%;
    }}

    #{container_id} .truncate:hover::after {{
        content: attr(data-full-text);
        position: absolute;
        left: 0;
        top: 100%;
        background: #1f2937;
        color: white;
        padding: 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        white-space: normal;
        max-width: 300px;
        z-index: 1000;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}

    #{container_id} .btn {{
        padding: 0.09375rem 0.1875rem;
        border-radius: 0.25rem;
        font-size: 0.6875rem;
        border: none;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 0.125rem;
        transition: all 0.15s;
    }}

    #{container_id} .btn:hover {{
        opacity: 0.85;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }}

    #{container_id} .btn-blue {{
        background: #dbeafe;
        color: #3b82f6;
    }}

    #{container_id} .btn-green {{
        background: #d1fae5;
        color: #16a34a;
    }}

    #{container_id} .btn-purple {{
        background: #e9d5ff;
        color: #a855f7;
    }}

    #{container_id} .btn-red {{
        background: #fee2e2;
        color: #ef4444;
    }}

    #{container_id} .btn-gray {{
        background: #f3f4f6;
        color: #6b7280;
    }}

    #{container_id} .type-badge {{
        display: inline-block;
        padding: 0.125rem 0.375rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        font-weight: 500;
        background: #ffffff;
        color: #374151;
        text-align: center;
        white-space: nowrap;
        border: 1px solid #d1d5db;
    }}

    #{container_id} .health-status {{
        display: flex;
        align-items: center;
        gap: 0.25rem;
        font-size: 0.75rem;
    }}

    #{container_id} .health-online {{
        color: #16a34a;
    }}

    #{container_id} .health-offline {{
        color: #ef4444;
    }}

    #{container_id} .health-timeout {{
        color: #f59e0b;
    }}

    #{container_id} .health-unknown {{
        color: #6b7280;
    }}

    #{container_id} .pricing {{
        font-weight: 500;
        color: #059669;
    }}

    #{container_id} .pricing.free {{
        color: #16a34a;
    }}

    #{container_id} .pricing.paid {{
        color: #dc2626;
    }}

    /* Better tags management */
    #{container_id} .tags-container {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.25rem;
        max-height: 2.5rem;
        overflow: hidden;
    }}

    #{container_id} .tags-more {{
        background: #e5e7eb;
        color: #6b7280;
        padding: 0.125rem 0.375rem;
        border-radius: 0.25rem;
        font-size: 0.625rem;
        border: 1px solid #d1d5db;
        cursor: pointer;
        transition: all 0.15s;
    }}

    #{container_id} .tags-more:hover {{
        background: #d1d5db;
        color: #374151;
    }}

    #{container_id} .summary {{
        color: #6b7280;
        font-style: italic;
        max-width: 200px;
    }}
    </style>
</head>
<body>
    <!-- Loading container -->
    <div id="loading-container-{container_id}" style="height: 100vh; display: flex; flex-direction: column; justify-content: center; align-items: center; background-color: #ffffff;">
        <!-- SyftBox Logo -->
        <svg class="syftbox-logo" width="120" height="139" viewBox="0 0 311 360" fill="none" xmlns="http://www.w3.org/2000/svg">
            <g clip-path="url(#clip0_7523_4240)">
                <path d="M311.414 89.7878L155.518 179.998L-0.378906 89.7878L155.518 -0.422485L311.414 89.7878Z" fill="url(#paint0_linear_7523_4240)"></path>
                <path d="M311.414 89.7878V270.208L155.518 360.423V179.998L311.414 89.7878Z" fill="url(#paint1_linear_7523_4240)"></path>
                <path d="M155.518 179.998V360.423L-0.378906 270.208V89.7878L155.518 179.998Z" fill="url(#paint2_linear_7523_4240)"></path>
            </g>
            <defs>
                <linearGradient id="paint0_linear_7523_4240" x1="-0.378904" y1="89.7878" x2="311.414" y2="89.7878" gradientUnits="userSpaceOnUse">
                    <stop stop-color="#DC7A6E"></stop>
                    <stop offset="0.251496" stop-color="#F6A464"></stop>
                    <stop offset="0.501247" stop-color="#FDC577"></stop>
                    <stop offset="0.753655" stop-color="#EFC381"></stop>
                    <stop offset="1" stop-color="#B9D599"></stop>
                </linearGradient>
                <linearGradient id="paint1_linear_7523_4240" x1="309.51" y1="89.7878" x2="155.275" y2="360.285" gradientUnits="userSpaceOnUse">
                    <stop stop-color="#BFCD94"></stop>
                    <stop offset="0.245025" stop-color="#B2D69E"></stop>
                    <stop offset="0.504453" stop-color="#8DCCA6"></stop>
                    <stop offset="0.745734" stop-color="#5CB8B7"></stop>
                    <stop offset="1" stop-color="#4CA5B8"></stop>
                </linearGradient>
                <linearGradient id="paint2_linear_7523_4240" x1="-0.378906" y1="89.7878" 
                               x2="155.761" y2="360.282" gradientUnits="userSpaceOnUse">
                    <stop stop-color="#D7686D"></stop>
                    <stop offset="0.225" stop-color="#C64B77"></stop>
                    <stop offset="0.485" stop-color="#A2638E"></stop>
                    <stop offset="0.703194" stop-color="#758AA8"></stop>
                    <stop offset="1" stop-color="#639EAF"></stop>
                </linearGradient>
                <clipPath id="clip0_7523_4240">
                    <rect width="311" height="360" fill="white"></rect>
                </clipPath>
            </defs>
        </svg>

        <div style="font-size: 20px; font-weight: 600; color: #000000; 
                    margin-top: 2rem; text-align: center;">
            loading <br />AI models
        </div>

        <div style="width: 340px; height: 6px; 
                    background-color: #e5e5e5; 
                    border-radius: 3px; margin: 1.5rem auto; overflow: hidden;">
            <div id="loading-bar-{container_id}" class="progress-bar-gradient" 
                 style="width: 0%; height: 100%;"></div>
        </div>

        <div id="loading-status-{container_id}" 
             style="color: #9ca3af; font-size: 0.875rem; margin-top: 0.5rem;">
            Initializing...
        </div>

        <div style="margin-top: 3rem; padding: 0 2rem; max-width: 500px; text-align: center;">
            <div style="color: #6b7280; font-size: 0.875rem; font-style: italic;">
                💡 Tip: {footer_tip}
            </div>
        </div>
    </div>

    <!-- Main widget container (hidden initially) -->
    <div id="{container_id}" style="display: none;">
        <div class="search-controls">
            <input id="{container_id}-search" placeholder="🔍 Search models..." style="flex: 1;">
            <input id="{container_id}-owner-filter" placeholder="Filter by Owner..." style="flex: 1;">
            <select id="{container_id}-service-filter" style="flex: 1;">
                <option value="">All Services</option>
                <option value="chat">Chat Only</option>
                <option value="search">Search Only</option>
            </select>
            <select id="{container_id}-pricing-filter" style="flex: 1;">
                <option value="">All Pricing</option>
                <option value="free">Free Only</option>
                <option value="paid">Paid Only</option>
            </select>
        </div>

        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th style="width: 15%;">Model Name</th>
                        <th style="width: 12%;">Owner</th>
                        <th style="width: 12%;">Services</th>
                        <th style="width: 10%;">Pricing</th>
                        <th style="width: 10%;">Status</th>
                        <th style="width: 15%;">Tags</th>
                        <th style="width: 26%;">Summary</th>
                    </tr>
                </thead>
                <tbody id="{container_id}-tbody">
                    <!-- Table rows will be populated by JavaScript -->
                </tbody>
            </table>
        </div>

        <div class="pagination">
            <div>
                <a href="https://github.com/OpenMined/syft-nsai-sdk/issues" target="_blank" style="color: #6b7280; text-decoration: none; font-size: 0.75rem;">
                    Report a Bug
                </a>
            </div>
            <span class="status" id="{container_id}-status">Loading...</span>
            <div class="pagination-controls">
                <button onclick="changePage_{container_id}(-1)" id="{container_id}-prev-btn" disabled>Previous</button>
                <span class="page-info" id="{container_id}-page-info">Page 1 of 1</span>
                <button onclick="changePage_{container_id}(1)" id="{container_id}-next-btn">Next</button>
            </div>
        </div>
    </div>

    <script>
    (function() {{
        // Configuration
        var CONFIG = {{
            currentUserEmail: '{current_user_email}'
        }};

        // Initialize variables
        var allModels = [];
        var filteredModels = [];
        var currentPage = {page};
        var itemsPerPage = {items_per_page};

        // Update progress
        function updateProgress(percent, status) {{
            var loadingBar = document.getElementById('loading-bar-{container_id}');
            var loadingStatus = document.getElementById('loading-status-{container_id}');

            if (loadingBar) {{
                loadingBar.style.width = percent + '%';
            }}
            if (loadingStatus) {{
                loadingStatus.innerHTML = status;
            }}
        }}

        // Load models data
        async function loadModels() {{
            try {{
                updateProgress(10, 'Initializing...');
                
                // Use the actual models data passed from Python
                var realModels = {json.dumps(models) if models else '[]'};
                
                if (realModels && realModels.length > 0) {{
                    // Use real models data
                    allModels = realModels;
                    updateProgress(100, 'Models loaded successfully!');
                }} else {{
                    // No models found
                    allModels = [];
                    updateProgress(100, 'No models found');
                }}
                
                filteredModels = allModels.slice();
                
                // Hide loading screen and show widget
                document.getElementById('loading-container-{container_id}').style.display = 'none';
                document.getElementById('{container_id}').style.display = 'flex';
                
                // Initial render
                renderTable();
                updateStatus();
                
            }} catch (error) {{
                console.error('Error loading models:', error);
                updateProgress(0, 'Error loading models. Please refresh the page.');
            }}
        }}



        // Render table
        function renderTable() {{
            var tbody = document.getElementById('{container_id}-tbody');
            var totalModels = filteredModels.length;
            var totalPages = Math.max(1, Math.ceil(totalModels / itemsPerPage));

            if (currentPage > totalPages) currentPage = totalPages;
            if (currentPage < 1) currentPage = 1;

            document.getElementById('{container_id}-prev-btn').disabled = currentPage === 1;
            document.getElementById('{container_id}-next-btn').disabled = currentPage === totalPages;
            document.getElementById('{container_id}-page-info').textContent = 'Page ' + currentPage + ' of ' + totalPages;

            if (totalModels === 0) {{
                tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; padding: 40px;">No models found</td></tr>';
                return;
            }}

            var start = (currentPage - 1) * itemsPerPage;
            var end = Math.min(start + itemsPerPage, totalModels);

            var html = '';
            for (var i = start; i < end; i++) {{
                var model = filteredModels[i];
                
                html += '<tr onclick="copyModelId_{container_id}(\\'' + model.name + '\\', \\'' + model.owner + '\\')">' +
                    '<td><div class="truncate" data-full-text="' + escapeHtml(model.name) + '" title="' + escapeHtml(model.name) + '">' + escapeHtml(model.name) + '</div></td>' +
                    '<td><div class="truncate" data-full-text="' + escapeHtml(model.owner) + '" title="' + escapeHtml(model.owner) + '">' + escapeHtml(model.owner) + '</div></td>' +
                    '<td>' + formatServices(model.services) + '</td>' +
                    '<td>' + formatPricing(model.min_pricing, model.max_pricing) + '</td>' +
                    '<td>' + formatStatus(model.config_status, model.health_status) + '</td>' +
                    '<td>' + formatTags(model.tags) + '</td>' +
                    '<td><div class="summary truncate" data-full-text="' + escapeHtml(model.summary) + '" title="' + escapeHtml(model.summary) + '">' + escapeHtml(model.summary) + '</div></td>' +
                '</tr>';
            }}

            tbody.innerHTML = html;
        }}

        // Format services
        function formatServices(services) {{
            if (!services || services.length === 0) return '<span class="type-badge">None</span>';
            
            var enabledServices = services.filter(s => s.enabled);
            if (enabledServices.length === 0) return '<span class="type-badge">Disabled</span>';
            
            return enabledServices.map(s => 
                '<span class="type-badge">' + s.type + '</span>'
            ).join(' ');
        }}

        // Format pricing
        function formatPricing(minPrice, maxPrice) {{
            if (minPrice === 0 && maxPrice === 0) {{
                return '<span class="pricing free">Free</span>';
            }} else if (minPrice === maxPrice) {{
                return '<span class="pricing paid">$' + minPrice.toFixed(3) + '</span>';
            }} else {{
                return '<span class="pricing paid">$' + minPrice.toFixed(3) + ' - $' + maxPrice.toFixed(3) + '</span>';
            }}
        }}

        // Format status
        function formatStatus(configStatus, healthStatus) {{
            var statusHtml = '<span class="type-badge">' + configStatus + '</span>';
            
            if (healthStatus) {{
                var healthClass = 'health-' + healthStatus;
                var healthIcon = '';
                
                switch(healthStatus) {{
                    case 'online': healthIcon = '✅'; break;
                    case 'offline': healthIcon = '❌'; break;
                    case 'timeout': healthIcon = '⏱️'; break;
                    default: healthIcon = '❓';
                }}
                
                statusHtml += ' <span class="' + healthClass + '">' + healthIcon + '</span>';
            }}
            
            return statusHtml;
        }}

        // Format tags
        function formatTags(tags) {{
            if (!tags || tags.length === 0) return '<span style="color: #9ca3af;">None</span>';
            
            var visibleTags = tags.slice(0, 2); // Show only 2 tags
            var remainingCount = tags.length - 2;
            
            var html = '<div class="tags-container">';
            
            // Add visible tags
            visibleTags.forEach(tag => {{
                html += '<span class="tag" title="' + escapeHtml(tag) + '">' + escapeHtml(tag) + '</span>';
            }});
            
            // Add "more" indicator if there are additional tags
            if (remainingCount > 0) {{
                html += '<span class="tags-more" onclick="showAllTags_{container_id}(event, ' + JSON.stringify(tags) + ')" title="Click to see all ' + tags.length + ' tags">+' + remainingCount + '</span>';
            }}
            
            html += '</div>';
            return html;
        }}

        // Function to show all tags in a popup
        function showAllTags_{container_id}(event, tags) {{
            event.stopPropagation();
            
            var popup = document.createElement('div');
            popup.style.cssText = `
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: white;
                border: 1px solid #d1d5db;
                border-radius: 0.5rem;
                padding: 1rem;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
                z-index: 1000;
                max-width: 400px;
                max-height: 300px;
                overflow-y: auto;
            `;
            
            popup.innerHTML = `
                <h3 style="margin: 0 0 0.5rem 0; font-size: 0.875rem;">All Tags (${{tags.length}})</h3>
                <div style="display: flex; flex-wrap: wrap; gap: 0.25rem;">
                    ${{tags.map(tag => `<span class="tag">${{escapeHtml(tag)}}</span>`).join('')}}
                </div>
                <button onclick="this.parentElement.remove()" style="margin-top: 0.5rem; padding: 0.25rem 0.5rem; border: 1px solid #d1d5db; border-radius: 0.25rem; background: white; cursor: pointer;">Close</button>
            `;
            
            document.body.appendChild(popup);
            
            // Close popup when clicking outside
            setTimeout(() => {{
                document.addEventListener('click', function closePopup(e) {{
                    if (!popup.contains(e.target)) {{
                        popup.remove();
                        document.removeEventListener('click', closePopup);
                    }}
                }});
            }}, 100);
        }}

        // Update status
        function updateStatus() {{
            var modelCount = filteredModels.length;
            var chatModels = filteredModels.filter(m => m.services.some(s => s.type === 'chat' && s.enabled)).length;
            var searchModels = filteredModels.filter(m => m.services.some(s => s.type === 'search' && s.enabled)).length;
            var freeModels = filteredModels.filter(m => m.min_pricing === 0).length;
            var paidModels = filteredModels.filter(m => m.min_pricing > 0).length;
            
            var statusText = modelCount + ' models • ' + chatModels + ' chat • ' + searchModels + ' search • ' + freeModels + ' free • ' + paidModels + ' paid';
            
            if (showFooterTip) {{
                statusText += ' • 💡 ' + '{footer_tip}';
            }}
            
            document.getElementById('{container_id}-status').textContent = statusText;
        }}

        // Search models
        function searchModels_{container_id}() {{
            var searchTerm = document.getElementById('{container_id}-search').value.toLowerCase();
            var ownerFilter = document.getElementById('{container_id}-owner-filter').value.toLowerCase();
            var serviceFilter = document.getElementById('{container_id}-service-filter').value;
            var pricingFilter = document.getElementById('{container_id}-pricing-filter').value;

            filteredModels = allModels.filter(function(model) {{
                // Owner filter
                if (ownerFilter && !model.owner.toLowerCase().includes(ownerFilter)) {{
                    return false;
                }}
                
                // Service filter
                if (serviceFilter) {{
                    var hasService = model.services.some(s => s.type === serviceFilter && s.enabled);
                    if (!hasService) return false;
                }}
                
                // Pricing filter
                if (pricingFilter === 'free' && model.min_pricing > 0) {{
                    return false;
                }} else if (pricingFilter === 'paid' && model.min_pricing === 0) {{
                    return false;
                }}
                
                // Search filter
                if (searchTerm) {{
                    var searchableContent = [
                        model.name,
                        model.owner,
                        model.summary,
                        model.description,
                        model.tags.join(' ')
                    ].join(' ').toLowerCase();
                    
                    return searchableContent.includes(searchTerm);
                }}
                
                return true;
            }});

            currentPage = 1;
            renderTable();
            updateStatus();
        }}

        // Change page
        function changePage_{container_id}(direction) {{
            var totalPages = Math.max(1, Math.ceil(filteredModels.length / itemsPerPage));
            currentPage += direction;
            if (currentPage < 1) currentPage = 1;
            if (currentPage > totalPages) currentPage = totalPages;
            renderTable();
        }}

        // Copy model ID
        function copyModelId_{container_id}(name, owner) {{
            var modelId = name + ' by ' + owner;
            navigator.clipboard.writeText(modelId).then(function() {{
                document.getElementById('{container_id}-status').textContent = 'Copied: ' + modelId;
                setTimeout(function() {{
                    updateStatus();
                }}, 2000);
            }});
        }}

        // Chat with model
        function chatWithModel_{container_id}(name, owner) {{
            var command = 'await client.chat(model_name="' + name + '", owner="' + owner + '", prompt="Hello!")';
            navigator.clipboard.writeText(command).then(function() {{
                document.getElementById('{container_id}-status').textContent = 'Chat command copied to clipboard';
                setTimeout(function() {{
                    updateStatus();
                }}, 2000);
            }});
        }}

        // Search with model
        function searchWithModel_{container_id}(name, owner) {{
            var command = 'await client.search(model_name="' + name + '", owner="' + owner + '", query="search query")';
            navigator.clipboard.writeText(command).then(function() {{
                document.getElementById('{container_id}-status').textContent = 'Search command copied to clipboard';
                setTimeout(function() {{
                    updateStatus();
                }}, 2000);
            }});
        }}

        // Refresh models
        function refreshModels_{container_id}() {{
            document.getElementById('{container_id}-status').textContent = 'Refreshing models...';
            loadModels();
        }}

        // Utility functions
        function escapeHtml(text) {{
            var div = document.createElement('div');
            div.textContent = text || '';
            return div.innerHTML;
        }}

        // Add event listeners
        document.getElementById('{container_id}-search').addEventListener('input', searchModels_{container_id});
        document.getElementById('{container_id}-owner-filter').addEventListener('input', searchModels_{container_id});
        document.getElementById('{container_id}-service-filter').addEventListener('change', searchModels_{container_id});
        document.getElementById('{container_id}-pricing-filter').addEventListener('change', searchModels_{container_id});

        // Start loading models when page loads
        loadModels();
    }})();
    </script>
</body>
</html>
"""
