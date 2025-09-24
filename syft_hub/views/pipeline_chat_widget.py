
"""
Interactive chat widget for Pipeline objects.
Provides a chat interface that uses the pipeline's data sources and synthesizer.
"""
import json
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.pipeline import Pipeline

def get_pipeline_chat_widget_html(pipeline: 'Pipeline', pipeline_var_name: str = 'pipeline') -> str:
    """Generate interactive chat widget HTML for a pipeline."""
    
    container_id = f"pipeline_chat_{uuid.uuid4().hex[:8]}"
    
    # Get pipeline configuration
    data_sources = [{"name": spec.name, "params": spec.params} for spec in pipeline.data_sources]
    synthesizer = {"name": pipeline.synthesizer.name, "params": pipeline.synthesizer.params} if pipeline.synthesizer else None
    
    # Check if pipeline is configured
    is_configured = bool(pipeline.data_sources and pipeline.synthesizer)
    
    # Estimate cost per message
    try:
        estimated_cost = pipeline.estimate_cost(1)
        cost_display = f"${estimated_cost:.4f}" if estimated_cost > 0 else "Free"
    except:
        cost_display = "Unable to calculate"
    
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Chat</title>
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
        max-width: 800px;
        margin: 0 auto;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        height: 600px;
        display: flex;
        flex-direction: column;
    }}
    
    .chat-header {{
        background: #f7f7f7;
        padding: 16px 20px;
        border-bottom: 1px solid #e1e1e1;
        flex-shrink: 0;
    }}
    
    .chat-title {{
        margin: 0 0 4px 0;
        font-size: 16px;
        font-weight: 600;
        color: #1a1a1a;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    
    .chat-status {{
        display: inline-block;
        padding: 2px 6px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 500;
        background: {'#d4edda' if is_configured else '#fff3cd'};
        color: {'#155724' if is_configured else '#856404'};
    }}
    
    .chat-config {{
        font-size: 12px;
        color: #666;
        margin: 4px 0 0 0;
    }}
    
    .chat-messages {{
        flex: 1;
        overflow-y: auto;
        padding: 16px;
        background: #fefefe;
        min-height: 0;
    }}
    
    .message {{
        margin-bottom: 16px;
        display: flex;
        align-items: flex-start;
        gap: 8px;
    }}
    
    .message.user {{
        flex-direction: row-reverse;
    }}
    
    .message-avatar {{
        width: 28px;
        height: 28px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: 500;
        flex-shrink: 0;
    }}
    
    .message.user .message-avatar {{
        background: #007bff;
        color: white;
    }}
    
    .message.assistant .message-avatar {{
        background: #6c757d;
        color: white;
    }}
    
    .message-content {{
        max-width: 70%;
        padding: 10px 14px;
        border-radius: 16px;
        font-size: 14px;
        line-height: 1.4;
    }}
    
    .message.user .message-content {{
        background: #007bff;
        color: white;
        border-bottom-right-radius: 4px;
    }}
    
    .message.assistant .message-content {{
        background: #f1f3f5;
        color: #333;
        border-bottom-left-radius: 4px;
    }}
    
    .message-info {{
        font-size: 11px;
        color: #666;
        margin-top: 4px;
        text-align: right;
    }}
    
    .message.assistant .message-info {{
        text-align: left;
    }}
    
    .chat-input {{
        padding: 16px;
        border-top: 1px solid #e1e1e1;
        background: #f9f9f9;
        flex-shrink: 0;
    }}
    
    .input-container {{
        display: flex;
        gap: 8px;
        align-items: flex-end;
    }}
    
    .chat-textarea {{
        flex: 1;
        min-height: 20px;
        max-height: 100px;
        padding: 10px 12px;
        border: 1px solid #d0d0d0;
        border-radius: 20px;
        font-size: 14px;
        font-family: inherit;
        resize: none;
        overflow-y: auto;
        background: white;
        line-height: 1.4;
    }}
    
    .chat-textarea:focus {{
        outline: none;
        border-color: #007bff;
        box-shadow: 0 0 0 2px rgba(0,122,204,0.1);
    }}
    
    .send-button {{
        padding: 10px 16px;
        background: #007bff;
        color: white;
        border: none;
        border-radius: 20px;
        font-size: 14px;
        cursor: pointer;
        font-weight: 500;
        transition: background-color 0.2s;
        height: 40px;
    }}
    
    .send-button:hover:not(:disabled) {{
        background: #0056b3;
    }}
    
    .send-button:disabled {{
        background: #6c757d;
        cursor: not-allowed;
    }}
    
    .loading-message {{
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
        color: #666;
        font-style: italic;
    }}
    
    .loading-dots {{
        display: inline-block;
    }}
    
    .loading-dots::after {{
        content: '';
        animation: loading 1.4s infinite;
    }}
    
    @keyframes loading {{
        0%, 20% {{ content: '.'; }}
        40% {{ content: '..'; }}
        60% {{ content: '...'; }}
        80%, 100% {{ content: ''; }}
    }}
    
    .error-message {{
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 16px;
        font-size: 13px;
    }}
    
    .welcome-message {{
        text-align: center;
        color: #666;
        font-style: italic;
        margin: 40px 20px;
        padding: 20px;
        border: 1px dashed #d0d0d0;
        border-radius: 8px;
        background: #f8f9fa;
    }}
    
    .welcome-title {{
        font-size: 16px;
        font-weight: 500;
        color: #495057;
        margin-bottom: 8px;
    }}
    
    .config-not-ready {{
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 16px;
        margin: 20px;
        text-align: center;
    }}
    </style>
</head>
<body>
    <div id="{container_id}">
        <div class="chat-header">
            <div class="chat-title">
                🤖 Pipeline Chat <span class="chat-status">{'Ready' if is_configured else 'Not Configured'}</span>
            </div>
            <div class="chat-config">
                {len(pipeline.data_sources)} data source(s) • {'1 synthesizer' if pipeline.synthesizer else 'No synthesizer'} • Cost: {cost_display}/msg
            </div>
        </div>
        
        <div class="chat-messages" id="{container_id}-messages">
            {'<div class="welcome-message"><div class="welcome-title">Welcome to Pipeline Chat!</div>Ask questions and get responses powered by your configured data sources and AI model.</div>' if is_configured else '<div class="config-not-ready">⚠️ Pipeline not fully configured<br><br>Please add data sources and a synthesizer before chatting.</div>'}
        </div>
        
        <div class="chat-input">
            <div class="input-container">
                <textarea 
                    class="chat-textarea" 
                    id="{container_id}-input" 
                    placeholder="{'Type your message...' if is_configured else 'Configure pipeline first...'}"
                    {'disabled' if not is_configured else ''}
                    rows="1"
                ></textarea>
                <button 
                    class="send-button" 
                    id="{container_id}-send"
                    onclick="sendMessage_{container_id}()"
                    {'disabled' if not is_configured else ''}
                >
                    Send
                </button>
            </div>
        </div>
    </div>

    <script>
    (function() {{
        const widgetId = '{container_id}';
        
        // Configuration from pipeline
        const pipelineConfig = {{
            data_sources: {json.dumps(data_sources)},
            synthesizer: {json.dumps(synthesizer)},
            is_configured: {json.dumps(is_configured)},
            cost_per_message: "{cost_display}"
        }};
        
        let messageHistory = [];
        let isProcessing = false;
        
        const messagesContainer = document.getElementById('{container_id}-messages');
        const inputElement = document.getElementById('{container_id}-input');
        const sendButton = document.getElementById('{container_id}-send');
        
        // Auto-resize textarea
        function autoResize(textarea) {{
            textarea.style.height = 'auto';
            textarea.style.height = Math.min(textarea.scrollHeight, 100) + 'px';
        }}
        
        inputElement.addEventListener('input', function() {{
            autoResize(this);
        }});
        
        // Handle Enter key (send message) and Shift+Enter (new line)
        inputElement.addEventListener('keydown', function(e) {{
            if (e.key === 'Enter' && !e.shiftKey) {{
                e.preventDefault();
                if (!isProcessing && pipelineConfig.is_configured) {{
                    sendMessage();
                }}
            }}
        }});
        
        function addMessage(content, type, info = null) {{
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${{type}}`;
            
            const avatar = type === 'user' ? 'U' : 'AI';
            const avatarColor = type === 'user' ? '#007bff' : '#6c757d';
            
            messageDiv.innerHTML = `
                <div class="message-avatar">${{avatar}}</div>
                <div class="message-content">
                    ${{content.replace(/\\n/g, '<br>')}}
                    ${{info ? `<div class="message-info">${{info}}</div>` : ''}}
                </div>
            `;
            
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }}
        
        function addLoadingMessage() {{
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'loading-message';
            loadingDiv.id = '{container_id}-loading';
            loadingDiv.innerHTML = '🔍 Processing through pipeline<span class="loading-dots"></span>';
            
            messagesContainer.appendChild(loadingDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }}
        
        function removeLoadingMessage() {{
            const loading = document.getElementById('{container_id}-loading');
            if (loading) {{
                loading.remove();
            }}
        }}
        
        function addErrorMessage(error) {{
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error-message';
            errorDiv.innerHTML = `❌ Error: ${{error}}`;
            
            messagesContainer.appendChild(errorDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }}
        
        async function sendMessage() {{
            const message = inputElement.value.trim();
            if (!message || isProcessing) return;
            
            // Add user message
            addMessage(message, 'user');
            messageHistory.push({{"role": "user", "content": message}});
            
            // Clear input and disable controls
            inputElement.value = '';
            autoResize(inputElement);
            isProcessing = true;
            sendButton.disabled = true;
            inputElement.disabled = true;
            
            // Add loading indicator
            addLoadingMessage();
            
            try {{
                // Execute the actual pipeline
                const response = await executePipeline(message);
                
                removeLoadingMessage();
                
                if (!response.success || response.error) {{
                    addErrorMessage(response.error || 'Unknown error occurred');
                }} else {{
                    const costInfo = response.cost > 0 ? `Cost: ${{response.cost.toFixed(4)}}` : 'Free';
                    const searchInfo = response.search_results > 0 ? ` • ${{response.search_results}} results found` : '';
                    addMessage(response.content, 'assistant', costInfo + searchInfo);
                    messageHistory.push({{"role": "assistant", "content": response.content}});
                }}
                
            }} catch (error) {{
                removeLoadingMessage();
                addErrorMessage(error.message || 'Unknown error occurred');
            }} finally {{
                // Re-enable controls
                isProcessing = false;
                sendButton.disabled = false;
                inputElement.disabled = false;
                inputElement.focus();
            }}
        }}
        
        // Execute pipeline - calls the actual pipeline.run method via kernel
        async function executePipeline(message) {{
            // Create a unique callback name for this execution
            const callbackName = `pipelineCallback_${{Math.random().toString(36).substr(2, 9)}}`;
            
            // Store the callback function
            window[callbackName] = function(result) {{
                // Clean up the callback
                delete window[callbackName];
                
                // Resolve with the result
                window.pipelinePromiseResolve(result);
            }};
            
            // Create a promise that will be resolved by the callback
            const resultPromise = new Promise((resolve, reject) => {{
                window.pipelinePromiseResolve = resolve;
                window.pipelinePromiseReject = reject;
                
                // Set a timeout in case something goes wrong
                setTimeout(() => {{
                    delete window[callbackName];
                    reject(new Error('Pipeline execution timeout'));
                }}, 30000); // 30 second timeout
            }});
            
            // Execute Python code to run the pipeline
            const pythonCode = `
import json
from IPython.display import Javascript

try:
    # Get the pipeline instance
    if '{pipeline_var_name}' in globals():
        pipeline_obj = globals()['{pipeline_var_name}']
        result = pipeline_obj.run([{{"role": "user", "content": "${{message.replace(/"/g, '\\"')}}"}}])
        
        # Extract result data
        response_content = result.response.message.content if result.response and result.response.message else "No response generated"
        cost = getattr(result, 'cost', 0.0)
        search_count = len(getattr(result, 'search_results', []))
        
        result_data = {{
            "content": response_content,
            "cost": cost,
            "search_results": search_count,
            "success": True
        }}
    else:
        result_data = {{
            "error": "Pipeline not found in global scope (looking for {pipeline_var_name})",
            "success": False
        }}
        
except Exception as e:
    result_data = {{
        "error": str(e),
        "success": False
    }}

# Return result via JavaScript callback
js_code = f"window.${{callbackName}}({{json.dumps(result_data)}});"
display(Javascript(js_code))
`;
            
            // Execute the Python code
            if (window.Jupyter && window.Jupyter.notebook) {{
                // In Jupyter notebook
                window.Jupyter.notebook.kernel.execute(pythonCode);
            }} else {{
                // Fallback - simulate execution
                setTimeout(() => {{
                    const sources = pipelineConfig.data_sources.map(s => s.name).join(', ');
                    const synthesizer = pipelineConfig.synthesizer ? pipelineConfig.synthesizer.name : 'None';
                    
                    window[callbackName]({{
                        content: `Simulated response for: "${{message}}"\\n\\nThis would use:\\n• Data sources: ${{sources}}\\n• Synthesizer: ${{synthesizer}}\\n\\nNote: Running in fallback mode - connect to Jupyter kernel for real pipeline execution.`,
                        cost: Math.random() * 0.01,
                        search_results: 2,
                        success: true
                    }});
                }}, 2000);
            }}
            
            return resultPromise;
        }}
        
        // Expose sendMessage function globally for button onclick
        window['sendMessage_{container_id}'] = sendMessage;
        
        // Focus on input when widget loads
        if (pipelineConfig.is_configured) {{
            inputElement.focus();
        }}
        
    }})();
    </script>
</body>
</html>
"""

class PipelineInteractiveWidget:
    """Interactive chat widget for Pipeline objects."""
    
    def __init__(self, pipeline: 'Pipeline'):
        self.pipeline = pipeline
        # Store pipeline in globals for JavaScript access
        import __main__
        setattr(__main__, f'_pipeline_{id(self)}', pipeline)
        self.pipeline_var_name = f'_pipeline_{id(self)}'
    
    def _is_jupyter_notebook(self) -> bool:
        """Detect if we're running in a Jupyter notebook environment."""
        try:
            import IPython
            ipython = IPython.get_ipython()
            if ipython is not None:
                if hasattr(ipython, 'kernel') and ipython.kernel is not None:
                    return True
                if 'ipykernel' in str(type(ipython)).lower():
                    return True
        except ImportError:
            pass
        return False
    
    def show(self) -> None:
        """Display the interactive chat widget."""
        if not self._is_jupyter_notebook():
            print("Interactive widget requires a Jupyter notebook environment.")
            print("Current pipeline configuration:")
            print(self.pipeline._get_text_representation())
            return
        
        try:
            from IPython.display import display, HTML
            html = get_pipeline_chat_widget_html(self.pipeline, self.pipeline_var_name)
            display(HTML(html))
            print("Interactive chat widget displayed")
        except ImportError:
            print("IPython not available. Interactive widget requires Jupyter notebook.")
        except Exception as e:
            print(f"Error displaying widget: {e}")
            import traceback
            traceback.print_exc()