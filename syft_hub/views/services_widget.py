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
) -> str:
    """Generate the services widget HTML for web serving."""
    
    container_id = f"syft_services_{uuid.uuid4().hex[:8]}"

    # Generate complete HTML with the widget
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SyftBox Services</title>
    <style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
        margin: 0;
        padding: 16px;
        background: #fafafa;
        color: #333;
        font-size: 13px;
        line-height: 1.5;
    }}
    
    #{container_id} {{
        border: 1px solid #e1e1e1;
        border-radius: 8px;
        background: #fff;
        max-width: 100%;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
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
        font-size: 13px;
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
        font-size: 13px;
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
        font-size: 12px;
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
        font-size: 12px;
    }}
    
    th {{
        background: #f6f6f6;
        padding: 10px;
        text-align: left;
        border-bottom: 1px solid #e1e1e1;
        font-weight: 500;
        font-size: 12px;
        color: #555;
        position: sticky;
        top: 0;
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
        background: #fff3e0;
        color: #ef6c00;
        border: 1px solid #ffcc02;
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
        margin-bottom: 3px;
        display: block;
        width: 100%;
        text-align: center;
        transition: all 0.15s ease;
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
        font-size: 13px;
    }}
    
    .pagination button {{
        padding: 6px 12px;
        border: 1px solid #c0c0c0;
        border-radius: 4px;
        background: #fff;
        color: #333;
        font-size: 12px;
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
    </style>
</head>
<body>
    <div id="{container_id}">
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
                        <th style="width: 15%">Name</th>
                        <th style="width: 15%">Datasite</th>
                        <th style="width: 12%">Type</th>
                        <th style="width: 10%">Pricing</th>
                        <th style="width: 10%">Availability</th>
                        <th style="width: 15%">Tags</th>
                        <th style="width: 18%">Description</th>
                        <th style="width: 5%">Copy</th>
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
</body>
</html>
"""