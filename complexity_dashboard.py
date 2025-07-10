#!/usr/bin/env python3
"""
Complexity Dashboard - Interactive visualization for Task Master AI complexity analysis

This module provides a web-based dashboard for visualizing task complexity metrics,
optimization opportunities, and execution planning results.
"""

import json
import time
import os
from typing import Dict, List, Any, Optional
from dataclasses import asdict
from task_complexity_analyzer import TaskComplexityAnalyzer, TaskComplexity
from optimization_engine import OptimizationEngine, ExecutionPlan, OptimizationStrategy
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import socketserver


class ComplexityDashboard:
    """
    Interactive dashboard for visualizing task complexity analysis and optimization results
    """
    
    def __init__(self, tasks_file: str = ".taskmaster/tasks/tasks.json"):
        """Initialize the dashboard"""
        self.tasks_file = tasks_file
        self.analyzer = TaskComplexityAnalyzer(tasks_file)
        self.optimizer = OptimizationEngine(self.analyzer, tasks_file)
        self.dashboard_dir = ".taskmaster/dashboard"
        self.port = 8080
        
    def generate_dashboard(self) -> str:
        """Generate complete HTML dashboard with interactive visualizations"""
        
        # Ensure dashboard directory exists
        os.makedirs(self.dashboard_dir, exist_ok=True)
        
        # Generate analysis data
        complexity_report = self.analyzer.generate_complexity_report()
        optimization_plan = self.optimizer.optimize_execution_order(OptimizationStrategy.ADAPTIVE_SCHEDULING)
        
        # Generate HTML content
        html_content = self._generate_html_dashboard(complexity_report, optimization_plan)
        
        # Save dashboard
        dashboard_file = os.path.join(self.dashboard_dir, "index.html")
        with open(dashboard_file, 'w') as f:
            f.write(html_content)
        
        # Generate supporting files
        self._generate_css_file()
        self._generate_js_file()
        self._generate_data_files(complexity_report, optimization_plan)
        
        return dashboard_file
    
    def _generate_html_dashboard(self, complexity_report: Dict[str, Any], 
                                optimization_plan: ExecutionPlan) -> str:
        """Generate the main HTML dashboard"""
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Task Master AI - Complexity Dashboard</title>
    <link rel="stylesheet" href="dashboard.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
</head>
<body>
    <div class="container">
        <header>
            <h1>🚀 Task Master AI - Complexity Dashboard</h1>
            <p class="subtitle">Advanced Task Analysis & Optimization Insights</p>
            <div class="timestamp">Generated: {complexity_report.get('analysis_timestamp', 'Unknown')}</div>
        </header>

        <div class="dashboard-grid">
            <!-- Summary Cards -->
            <div class="card summary-card">
                <h2>📊 Analysis Summary</h2>
                <div class="metric-grid">
                    <div class="metric">
                        <span class="metric-value">{complexity_report.get('summary', {}).get('total_tasks', 0)}</span>
                        <span class="metric-label">Total Tasks</span>
                    </div>
                    <div class="metric">
                        <span class="metric-value">{complexity_report.get('summary', {}).get('cpu_intensive_tasks', 0)}</span>
                        <span class="metric-label">CPU Intensive</span>
                    </div>
                    <div class="metric">
                        <span class="metric-value">{complexity_report.get('summary', {}).get('memory_intensive_tasks', 0)}</span>
                        <span class="metric-label">Memory Intensive</span>
                    </div>
                    <div class="metric">
                        <span class="metric-value">{complexity_report.get('summary', {}).get('highly_parallelizable_tasks', 0)}</span>
                        <span class="metric-label">Parallelizable</span>
                    </div>
                </div>
            </div>

            <!-- Complexity Distribution -->
            <div class="card chart-card">
                <h2>⚡ Complexity Distribution</h2>
                <canvas id="complexityChart"></canvas>
            </div>

            <!-- Resource Requirements -->
            <div class="card chart-card">
                <h2>💾 Resource Requirements</h2>
                <div class="resource-info">
                    <div class="resource-item">
                        <span class="resource-label">Total CPU Cores Needed:</span>
                        <span class="resource-value">{complexity_report.get('resource_requirements', {}).get('total_cpu_cores_needed', 0)}</span>
                    </div>
                    <div class="resource-item">
                        <span class="resource-label">Total Memory (GB):</span>
                        <span class="resource-value">{complexity_report.get('resource_requirements', {}).get('total_memory_gb_needed', 0):.1f}</span>
                    </div>
                    <div class="resource-item">
                        <span class="resource-label">Estimated Runtime (Hours):</span>
                        <span class="resource-value">{complexity_report.get('resource_requirements', {}).get('estimated_total_runtime_hours', 0):.2f}</span>
                    </div>
                </div>
                <canvas id="resourceChart"></canvas>
            </div>

            <!-- Optimization Results -->
            <div class="card optimization-card">
                <h2>🎯 Optimization Results</h2>
                <div class="optimization-info">
                    <div class="optimization-metric">
                        <span class="opt-label">Strategy:</span>
                        <span class="opt-value">{optimization_plan.strategy.value if optimization_plan else 'N/A'}</span>
                    </div>
                    <div class="optimization-metric">
                        <span class="opt-label">Efficiency Score:</span>
                        <span class="opt-value">{f"{optimization_plan.efficiency_score:.3f}" if optimization_plan else 'N/A'}</span>
                    </div>
                    <div class="optimization-metric">
                        <span class="opt-label">Parallel Groups:</span>
                        <span class="opt-value">{len(optimization_plan.parallel_groups) if optimization_plan else 0}</span>
                    </div>
                    <div class="optimization-metric">
                        <span class="opt-label">Optimized Runtime (Hours):</span>
                        <span class="opt-value">{f"{optimization_plan.estimated_total_time/3600:.2f}" if optimization_plan else 'N/A'}</span>
                    </div>
                </div>
            </div>

            <!-- Task Timeline -->
            <div class="card timeline-card">
                <h2>📅 Execution Timeline</h2>
                <div id="timeline-container"></div>
            </div>

            <!-- Bottlenecks & Recommendations -->
            <div class="card bottlenecks-card">
                <h2>⚠️ Bottlenecks & Issues</h2>
                <div class="bottlenecks-list">
                    {self._generate_bottlenecks_html(complexity_report, optimization_plan)}
                </div>
            </div>

            <!-- Recommendations -->
            <div class="card recommendations-card">
                <h2>💡 Optimization Recommendations</h2>
                <div class="recommendations-list">
                    {self._generate_recommendations_html(complexity_report)}
                </div>
            </div>

            <!-- Detailed Task Analysis -->
            <div class="card table-card">
                <h2>📋 Detailed Task Analysis</h2>
                <div class="table-container">
                    <table id="taskTable">
                        <thead>
                            <tr>
                                <th>Task ID</th>
                                <th>Time Complexity</th>
                                <th>Space Complexity</th>
                                <th>Runtime (s)</th>
                                <th>CPU Intensive</th>
                                <th>Memory Intensive</th>
                                <th>Parallelization</th>
                            </tr>
                        </thead>
                        <tbody>
                            {self._generate_task_table_html(complexity_report)}
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- System Resources -->
            <div class="card system-card">
                <h2>🖥️ System Resources</h2>
                <div class="system-info">
                    {self._generate_system_info_html(complexity_report)}
                </div>
            </div>
        </div>
    </div>

    <script src="dashboard.js"></script>
    <script>
        // Initialize dashboard with data
        const complexityData = {json.dumps(complexity_report.get('complexity_distribution', {}))};
        const optimizationData = {json.dumps(asdict(optimization_plan) if optimization_plan else {}, default=str)};
        
        // Initialize charts
        initializeComplexityChart(complexityData);
        initializeResourceChart({json.dumps(complexity_report.get('resource_requirements', {}))});
        initializeTimeline(optimizationData);
        
        // Make table sortable
        makeTableSortable('taskTable');
    </script>
</body>
</html>"""
    
    def _generate_bottlenecks_html(self, complexity_report: Dict[str, Any], 
                                  optimization_plan: ExecutionPlan) -> str:
        """Generate HTML for bottlenecks section"""
        
        bottlenecks = []
        
        # Add complexity report bottlenecks
        if 'optimization_opportunities' in complexity_report:
            bottlenecks.extend(complexity_report['optimization_opportunities'].get('resource_bottlenecks', []))
        
        # Add optimization plan bottlenecks
        if optimization_plan:
            bottlenecks.extend(optimization_plan.bottlenecks)
        
        if not bottlenecks:
            return '<div class="no-bottlenecks">✅ No significant bottlenecks identified</div>'
        
        html_items = []
        for bottleneck in bottlenecks:
            html_items.append(f'<div class="bottleneck-item">⚠️ {bottleneck}</div>')
        
        return '\n'.join(html_items)
    
    def _generate_recommendations_html(self, complexity_report: Dict[str, Any]) -> str:
        """Generate HTML for recommendations section"""
        
        recommendations = complexity_report.get('optimization_opportunities', {}).get('optimization_recommendations', [])
        
        if not recommendations:
            return '<div class="no-recommendations">✅ No specific recommendations at this time</div>'
        
        html_items = []
        for rec in recommendations:
            html_items.append(f'<div class="recommendation-item">💡 {rec}</div>')
        
        return '\n'.join(html_items)
    
    def _generate_task_table_html(self, complexity_report: Dict[str, Any]) -> str:
        """Generate HTML for detailed task analysis table"""
        
        detailed_analysis = complexity_report.get('detailed_analysis', [])
        
        if not detailed_analysis:
            return '<tr><td colspan="7">No task data available</td></tr>'
        
        html_rows = []
        for task in detailed_analysis:
            cpu_icon = "🔥" if task.get('cpu_intensive', False) else "❄️"
            memory_icon = "💾" if task.get('memory_intensive', False) else "📝"
            parallel_score = task.get('parallelization_potential', 0.0)
            parallel_bar = f'<div class="progress-bar"><div class="progress-fill" style="width: {parallel_score*100:.1f}%"></div></div>'
            
            html_rows.append(f"""
                <tr>
                    <td>{task.get('task_id', 'N/A')}</td>
                    <td>{task.get('time_complexity', 'N/A')}</td>
                    <td>{task.get('space_complexity', 'N/A')}</td>
                    <td>{task.get('estimated_runtime_seconds', 0):.1f}</td>
                    <td>{cpu_icon}</td>
                    <td>{memory_icon}</td>
                    <td>{parallel_bar}</td>
                </tr>
            """)
        
        return '\n'.join(html_rows)
    
    def _generate_system_info_html(self, complexity_report: Dict[str, Any]) -> str:
        """Generate HTML for system information"""
        
        system_resources = complexity_report.get('system_resources', {})
        
        return f"""
            <div class="system-metric">
                <span class="sys-label">CPU Cores:</span>
                <span class="sys-value">{system_resources.get('cpu_cores', 'N/A')}</span>
            </div>
            <div class="system-metric">
                <span class="sys-label">Available Memory:</span>
                <span class="sys-value">{system_resources.get('available_memory_gb', 0):.1f} GB</span>
            </div>
            <div class="system-metric">
                <span class="sys-label">Available Disk:</span>
                <span class="sys-value">{system_resources.get('available_disk_gb', 0):.1f} GB</span>
            </div>
            <div class="system-metric">
                <span class="sys-label">CPU Usage:</span>
                <span class="sys-value">{system_resources.get('cpu_usage_percent', 0):.1f}%</span>
            </div>
            <div class="system-metric">
                <span class="sys-label">Memory Usage:</span>
                <span class="sys-value">{system_resources.get('memory_usage_percent', 0):.1f}%</span>
            </div>
        """
    
    def _generate_css_file(self):
        """Generate CSS styling for the dashboard"""
        
        css_content = """
/* Task Master AI Dashboard Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    color: #333;
}

.container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
}

header {
    text-align: center;
    margin-bottom: 30px;
    color: white;
}

header h1 {
    font-size: 2.5em;
    margin-bottom: 10px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.subtitle {
    font-size: 1.2em;
    opacity: 0.9;
    margin-bottom: 10px;
}

.timestamp {
    font-size: 0.9em;
    opacity: 0.8;
}

.dashboard-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
    gap: 20px;
}

.card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.15);
}

.card h2 {
    margin-bottom: 15px;
    color: #4a5568;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 8px;
}

/* Summary Card */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 15px;
}

.metric {
    text-align: center;
    padding: 15px;
    background: #f7fafc;
    border-radius: 8px;
}

.metric-value {
    display: block;
    font-size: 2em;
    font-weight: bold;
    color: #2d3748;
}

.metric-label {
    font-size: 0.9em;
    color: #718096;
    margin-top: 5px;
}

/* Chart Cards */
.chart-card canvas {
    max-height: 300px;
}

/* Resource Info */
.resource-info {
    margin-bottom: 15px;
}

.resource-item {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
    padding: 5px 0;
    border-bottom: 1px solid #e2e8f0;
}

.resource-label {
    font-weight: 500;
    color: #4a5568;
}

.resource-value {
    font-weight: bold;
    color: #2d3748;
}

/* Optimization Card */
.optimization-info {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 15px;
}

.optimization-metric {
    padding: 10px;
    background: #f7fafc;
    border-radius: 6px;
}

.opt-label {
    display: block;
    font-size: 0.9em;
    color: #718096;
    margin-bottom: 4px;
}

.opt-value {
    font-weight: bold;
    color: #2d3748;
}

/* Timeline */
#timeline-container {
    min-height: 200px;
    background: #f7fafc;
    border-radius: 8px;
    padding: 15px;
}

/* Bottlenecks */
.bottleneck-item, .recommendation-item, .no-bottlenecks, .no-recommendations {
    padding: 10px;
    margin-bottom: 8px;
    border-radius: 6px;
    background: #fed7d7;
    border-left: 4px solid #e53e3e;
}

.recommendation-item {
    background: #c6f6d5;
    border-left-color: #38a169;
}

.no-bottlenecks, .no-recommendations {
    background: #d4edda;
    border-left-color: #28a745;
}

/* Table */
.table-container {
    overflow-x: auto;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
}

th, td {
    padding: 10px;
    text-align: left;
    border-bottom: 1px solid #e2e8f0;
}

th {
    background: #f7fafc;
    font-weight: 600;
    color: #4a5568;
    cursor: pointer;
}

th:hover {
    background: #edf2f7;
}

/* Progress Bar */
.progress-bar {
    width: 60px;
    height: 16px;
    background: #e2e8f0;
    border-radius: 8px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #48bb78, #38a169);
    transition: width 0.3s ease;
}

/* System Info */
.system-metric {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
    padding: 8px 0;
    border-bottom: 1px solid #e2e8f0;
}

.sys-label {
    color: #718096;
}

.sys-value {
    font-weight: bold;
    color: #2d3748;
}

/* Responsive */
@media (max-width: 768px) {
    .dashboard-grid {
        grid-template-columns: 1fr;
    }
    
    .metric-grid, .optimization-info {
        grid-template-columns: 1fr;
    }
    
    header h1 {
        font-size: 2em;
    }
}

/* Loading states */
.loading {
    opacity: 0.6;
    pointer-events: none;
}

.loading::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 20px;
    height: 20px;
    margin: -10px 0 0 -10px;
    border: 2px solid #ccc;
    border-top: 2px solid #333;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
"""
        
        css_file = os.path.join(self.dashboard_dir, "dashboard.css")
        with open(css_file, 'w') as f:
            f.write(css_content)
    
    def _generate_js_file(self):
        """Generate JavaScript for dashboard interactivity"""
        
        js_content = """
// Task Master AI Dashboard JavaScript

// Initialize complexity distribution chart
function initializeComplexityChart(data) {
    const ctx = document.getElementById('complexityChart').getContext('2d');
    
    const labels = Object.keys(data);
    const values = Object.values(data);
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: [
                    '#FF6384',
                    '#36A2EB',
                    '#FFCE56',
                    '#4BC0C0',
                    '#9966FF',
                    '#FF9F40',
                    '#FF6384',
                    '#C9CBCF'
                ],
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                title: {
                    display: true,
                    text: 'Task Complexity Distribution'
                }
            }
        }
    });
}

// Initialize resource requirements chart
function initializeResourceChart(data) {
    const ctx = document.getElementById('resourceChart').getContext('2d');
    
    const chartData = {
        labels: ['CPU Cores', 'Memory (GB)', 'Runtime (Hours)'],
        datasets: [{
            label: 'Resource Requirements',
            data: [
                data.total_cpu_cores_needed || 0,
                data.total_memory_gb_needed || 0,
                data.estimated_total_runtime_hours || 0
            ],
            backgroundColor: [
                'rgba(255, 99, 132, 0.7)',
                'rgba(54, 162, 235, 0.7)',
                'rgba(255, 206, 86, 0.7)'
            ],
            borderColor: [
                'rgba(255, 99, 132, 1)',
                'rgba(54, 162, 235, 1)',
                'rgba(255, 206, 86, 1)'
            ],
            borderWidth: 2
        }]
    };
    
    new Chart(ctx, {
        type: 'bar',
        data: chartData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Total Resource Requirements'
                }
            }
        }
    });
}

// Initialize timeline visualization
function initializeTimeline(optimizationData) {
    const container = document.getElementById('timeline-container');
    
    if (!optimizationData.parallel_groups || optimizationData.parallel_groups.length === 0) {
        container.innerHTML = '<p>No optimization data available for timeline visualization.</p>';
        return;
    }
    
    // Simple timeline visualization
    let timelineHTML = '<div class="timeline">';
    
    optimizationData.parallel_groups.forEach((group, index) => {
        timelineHTML += `
            <div class="timeline-group">
                <div class="timeline-header">Group ${index + 1}</div>
                <div class="timeline-tasks">
                    ${group.map(taskId => `<span class="timeline-task">Task ${taskId}</span>`).join('')}
                </div>
            </div>
        `;
    });
    
    timelineHTML += '</div>';
    
    // Add CSS for timeline
    timelineHTML += `
        <style>
        .timeline { margin-top: 15px; }
        .timeline-group { 
            margin-bottom: 15px; 
            border: 1px solid #e2e8f0; 
            border-radius: 6px; 
            padding: 10px; 
        }
        .timeline-header { 
            font-weight: bold; 
            color: #4a5568; 
            margin-bottom: 8px; 
        }
        .timeline-tasks { 
            display: flex; 
            flex-wrap: wrap; 
            gap: 5px; 
        }
        .timeline-task { 
            background: #e6fffa; 
            color: #234e52; 
            padding: 4px 8px; 
            border-radius: 4px; 
            font-size: 0.9em; 
        }
        </style>
    `;
    
    container.innerHTML = timelineHTML;
}

// Make table sortable
function makeTableSortable(tableId) {
    const table = document.getElementById(tableId);
    if (!table) return;
    
    const headers = table.querySelectorAll('th');
    
    headers.forEach((header, index) => {
        header.addEventListener('click', () => {
            sortTable(table, index);
        });
    });
}

function sortTable(table, columnIndex) {
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    
    const isNumeric = (str) => !isNaN(str) && !isNaN(parseFloat(str));
    
    // Determine sort direction
    const isAscending = !table.dataset.sortOrder || table.dataset.sortOrder === 'desc';
    table.dataset.sortOrder = isAscending ? 'asc' : 'desc';
    
    rows.sort((a, b) => {
        const aValue = a.cells[columnIndex].textContent.trim();
        const bValue = b.cells[columnIndex].textContent.trim();
        
        let comparison;
        if (isNumeric(aValue) && isNumeric(bValue)) {
            comparison = parseFloat(aValue) - parseFloat(bValue);
        } else {
            comparison = aValue.localeCompare(bValue);
        }
        
        return isAscending ? comparison : -comparison;
    });
    
    // Clear and re-append sorted rows
    tbody.innerHTML = '';
    rows.forEach(row => tbody.appendChild(row));
    
    // Update header indicators
    table.querySelectorAll('th').forEach(th => th.classList.remove('sort-asc', 'sort-desc'));
    table.querySelectorAll('th')[columnIndex].classList.add(isAscending ? 'sort-asc' : 'sort-desc');
}

// Auto-refresh functionality
function setupAutoRefresh() {
    const refreshInterval = 30000; // 30 seconds
    
    setInterval(() => {
        // In a real implementation, this would fetch fresh data
        console.log('Auto-refreshing dashboard data...');
    }, refreshInterval);
}

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    console.log('Task Master AI Dashboard loaded');
    setupAutoRefresh();
});

// Export functions for global access
window.TaskMasterDashboard = {
    initializeComplexityChart,
    initializeResourceChart,
    initializeTimeline,
    makeTableSortable
};
"""
        
        js_file = os.path.join(self.dashboard_dir, "dashboard.js")
        with open(js_file, 'w') as f:
            f.write(js_content)
    
    def _generate_data_files(self, complexity_report: Dict[str, Any], 
                            optimization_plan: ExecutionPlan):
        """Generate JSON data files for dashboard"""
        
        # Save complexity report
        complexity_file = os.path.join(self.dashboard_dir, "complexity_data.json")
        with open(complexity_file, 'w') as f:
            json.dump(complexity_report, f, indent=2, default=str)
        
        # Save optimization plan
        if optimization_plan:
            optimization_file = os.path.join(self.dashboard_dir, "optimization_data.json")
            with open(optimization_file, 'w') as f:
                json.dump(asdict(optimization_plan), f, indent=2, default=str)
    
    def start_server(self, port: int = None) -> str:
        """Start HTTP server for dashboard"""
        
        if port:
            self.port = port
        
        # Change to dashboard directory
        original_dir = os.getcwd()
        os.chdir(self.dashboard_dir)
        
        try:
            # Find available port
            while True:
                try:
                    with socketserver.TCPServer(("", self.port), SimpleHTTPRequestHandler) as httpd:
                        url = f"http://localhost:{self.port}"
                        print(f"Dashboard server starting at {url}")
                        
                        # Start server in background thread
                        server_thread = threading.Thread(target=httpd.serve_forever)
                        server_thread.daemon = True
                        server_thread.start()
                        
                        return url
                        
                except OSError:
                    self.port += 1
                    
        finally:
            os.chdir(original_dir)
    
    def launch_dashboard(self, auto_open: bool = True) -> str:
        """Generate and launch the complete dashboard"""
        
        print("Generating Task Master AI Complexity Dashboard...")
        
        # Generate dashboard files
        dashboard_file = self.generate_dashboard()
        
        # Start server
        url = self.start_server()
        
        print(f"Dashboard generated: {dashboard_file}")
        print(f"Server running at: {url}")
        
        if auto_open:
            print("Opening dashboard in browser...")
            webbrowser.open(url)
        
        return url


def main():
    """Main function for command-line usage"""
    import sys
    
    if len(sys.argv) > 1:
        tasks_file = sys.argv[1]
    else:
        tasks_file = ".taskmaster/tasks/tasks.json"
    
    dashboard = ComplexityDashboard(tasks_file)
    
    try:
        url = dashboard.launch_dashboard()
        print(f"\\nDashboard is running at: {url}")
        print("Press Ctrl+C to stop the server...")
        
        # Keep server running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\\nShutting down dashboard server...")


if __name__ == "__main__":
    main()