/**
 * Dashboard State Manager
 * Implements atomic task: Create frontend WebSocket integration and state management
 * 
 * Provides centralized state management for the Task-Master dashboard,
 * including WebSocket connection management, real-time data handling,
 * and reactive UI updates.
 */

class DashboardStateManager {
    constructor() {
        this.state = {
            // Connection state
            connection: {
                status: 'disconnected', // 'disconnected', 'connecting', 'connected'
                url: 'ws://localhost:8765',
                reconnectAttempts: 0,
                maxReconnectAttempts: 5,
                reconnectDelay: 1000,
                lastConnected: null,
                connectionStartTime: null
            },
            
            // Subscription management
            subscriptions: {
                telemetry: true,
                task_updates: true,
                system_metrics: true,
                health_status: true,
                heartbeat: true
            },
            
            // Real-time data
            data: {
                messages: [],
                maxMessages: 1000,
                telemetry: [],
                tasks: new Map(),
                systemMetrics: {},
                healthStatus: {},
                statistics: {
                    totalMessages: 0,
                    messagesByType: {},
                    connectionUptime: 0,
                    lastMessageTime: null
                }
            },
            
            // UI state
            ui: {
                selectedTab: 'overview',
                filters: {
                    messageType: 'all',
                    timeRange: '1h'
                },
                notifications: [],
                alerts: []
            }
        };
        
        this.subscribers = new Map();
        this.websocket = null;
        this.reconnectTimer = null;
        this.uptimeTimer = null;
        
        this.initializeEventHandlers();
    }
    
    /**
     * Subscribe to state changes
     */
    subscribe(path, callback) {
        if (!this.subscribers.has(path)) {
            this.subscribers.set(path, new Set());
        }
        this.subscribers.get(path).add(callback);
        
        // Return unsubscribe function
        return () => {
            const callbacks = this.subscribers.get(path);
            if (callbacks) {
                callbacks.delete(callback);
            }
        };
    }
    
    /**
     * Notify subscribers of state changes
     */
    notify(path, newValue, oldValue) {
        const callbacks = this.subscribers.get(path);
        if (callbacks) {
            callbacks.forEach(callback => {
                try {
                    callback(newValue, oldValue, path);
                } catch (error) {
                    console.error(`Error in state subscriber for ${path}:`, error);
                }
            });
        }
        
        // Notify wildcard subscribers
        const wildcardCallbacks = this.subscribers.get('*');
        if (wildcardCallbacks) {
            wildcardCallbacks.forEach(callback => {
                try {
                    callback(this.state, path);
                } catch (error) {
                    console.error('Error in wildcard state subscriber:', error);
                }
            });
        }
    }
    
    /**
     * Update state and notify subscribers
     */
    setState(path, value) {
        const pathParts = path.split('.');
        let current = this.state;
        
        // Navigate to parent object
        for (let i = 0; i < pathParts.length - 1; i++) {
            if (!current[pathParts[i]]) {
                current[pathParts[i]] = {};
            }
            current = current[pathParts[i]];
        }
        
        const lastKey = pathParts[pathParts.length - 1];
        const oldValue = current[lastKey];
        
        current[lastKey] = value;
        this.notify(path, value, oldValue);
    }
    
    /**
     * Get state value by path
     */
    getState(path) {
        const pathParts = path.split('.');
        let current = this.state;
        
        for (const part of pathParts) {
            if (current === null || current === undefined) {
                return undefined;
            }
            current = current[part];
        }
        
        return current;
    }
    
    /**
     * Initialize WebSocket connection
     */
    async connect() {
        if (this.state.connection.status === 'connected' || 
            this.state.connection.status === 'connecting') {
            return;
        }
        
        this.setState('connection.status', 'connecting');
        this.setState('connection.reconnectAttempts', 0);
        
        try {
            await this.establishWebSocketConnection();
        } catch (error) {
            console.error('Connection failed:', error);
            this.setState('connection.status', 'disconnected');
            this.addNotification('error', `Connection failed: ${error.message}`);
            
            // Schedule reconnection if not at max attempts
            if (this.state.connection.reconnectAttempts < this.state.connection.maxReconnectAttempts) {
                this.scheduleReconnect();
            }
        }
    }
    
    /**
     * Establish WebSocket connection
     */
    establishWebSocketConnection() {
        return new Promise((resolve, reject) => {
            try {
                this.websocket = new WebSocket(this.state.connection.url);
                
                this.websocket.onopen = () => {
                    this.setState('connection.status', 'connected');
                    this.setState('connection.connectionStartTime', new Date());
                    this.setState('connection.lastConnected', new Date());
                    this.setState('connection.reconnectAttempts', 0);
                    
                    this.addNotification('success', 'Connected to Task-Master WebSocket server');
                    this.startUptimeTimer();
                    resolve();
                };
                
                this.websocket.onmessage = (event) => {
                    this.handleWebSocketMessage(event);
                };
                
                this.websocket.onclose = (event) => {
                    this.handleWebSocketClose(event);
                };
                
                this.websocket.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    reject(new Error('WebSocket connection error'));
                };
                
                // Connection timeout
                setTimeout(() => {
                    if (this.websocket.readyState === WebSocket.CONNECTING) {
                        this.websocket.close();
                        reject(new Error('Connection timeout'));
                    }
                }, 10000);
                
            } catch (error) {
                reject(error);
            }
        });
    }
    
    /**
     * Disconnect WebSocket
     */
    disconnect() {
        if (this.websocket) {
            this.websocket.close();
        }
        
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        
        this.stopUptimeTimer();
        this.setState('connection.status', 'disconnected');
        this.setState('connection.connectionStartTime', null);
        
        this.addNotification('info', 'Disconnected from WebSocket server');
    }
    
    /**
     * Handle WebSocket message reception
     */
    handleWebSocketMessage(event) {
        try {
            const message = JSON.parse(event.data);
            this.processMessage(message);
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    }
    
    /**
     * Handle WebSocket connection close
     */
    handleWebSocketClose(event) {
        this.setState('connection.status', 'disconnected');
        this.stopUptimeTimer();
        
        if (!event.wasClean) {
            // Unexpected disconnection - schedule reconnect
            this.addNotification('warning', 'Connection lost. Attempting to reconnect...');
            this.scheduleReconnect();
        }
    }
    
    /**
     * Schedule reconnection attempt
     */
    scheduleReconnect() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
        }
        
        const attempts = this.state.connection.reconnectAttempts;
        if (attempts >= this.state.connection.maxReconnectAttempts) {
            this.addNotification('error', 'Max reconnection attempts reached');
            return;
        }
        
        const delay = this.state.connection.reconnectDelay * Math.pow(2, attempts);
        
        this.reconnectTimer = setTimeout(() => {
            this.setState('connection.reconnectAttempts', attempts + 1);
            this.connect();
        }, delay);
    }
    
    /**
     * Process incoming WebSocket message
     */
    processMessage(message) {
        // Update statistics
        const stats = this.state.data.statistics;
        stats.totalMessages++;
        stats.lastMessageTime = new Date();
        
        const messageType = message.type;
        if (!stats.messagesByType[messageType]) {
            stats.messagesByType[messageType] = 0;
        }
        stats.messagesByType[messageType]++;
        
        this.setState('data.statistics', { ...stats });
        
        // Add to message log
        this.addMessage(message);
        
        // Process by message type
        switch (messageType) {
            case 'telemetry':
                this.processTelemetryMessage(message);
                break;
            case 'task_update':
                this.processTaskUpdateMessage(message);
                break;
            case 'system_metric':
                this.processSystemMetricMessage(message);
                break;
            case 'health_status':
                this.processHealthStatusMessage(message);
                break;
            case 'heartbeat':
                this.processHeartbeatMessage(message);
                break;
            default:
                console.log('Unknown message type:', messageType);
        }
    }
    
    /**
     * Process telemetry message
     */
    processTelemetryMessage(message) {
        const telemetryData = {
            ...message.data,
            timestamp: message.timestamp,
            id: message.id
        };
        
        // Add to telemetry array (with size limit)
        const telemetry = [...this.state.data.telemetry, telemetryData];
        if (telemetry.length > 1000) {
            telemetry.shift(); // Remove oldest
        }
        
        this.setState('data.telemetry', telemetry);
    }
    
    /**
     * Process task update message
     */
    processTaskUpdateMessage(message) {
        const taskData = message.data;
        const taskId = taskData.task_id;
        
        // Update task in tasks map
        const tasks = new Map(this.state.data.tasks);
        const existingTask = tasks.get(taskId) || {};
        
        const updatedTask = {
            ...existingTask,
            ...taskData,
            lastUpdated: message.timestamp,
            messageId: message.id
        };
        
        tasks.set(taskId, updatedTask);
        this.setState('data.tasks', tasks);
        
        // Add alert for important task status changes
        if (taskData.status === 'completed') {
            this.addAlert('success', `Task ${taskId} completed`);
        } else if (taskData.status === 'failed') {
            this.addAlert('error', `Task ${taskId} failed`);
        }
    }
    
    /**
     * Process system metric message
     */
    processSystemMetricMessage(message) {
        const metricData = message.data;
        const metricName = metricData.metric_name;
        
        // Update system metrics
        const systemMetrics = { ...this.state.data.systemMetrics };
        systemMetrics[metricName] = {
            ...metricData,
            timestamp: message.timestamp,
            messageId: message.id
        };
        
        this.setState('data.systemMetrics', systemMetrics);
        
        // Check for alerts
        if (metricName === 'system.cpu.usage' && metricData.value > 90) {
            this.addAlert('warning', `High CPU usage: ${metricData.value.toFixed(1)}%`);
        }
    }
    
    /**
     * Process health status message
     */
    processHealthStatusMessage(message) {
        const healthData = {
            ...message.data,
            timestamp: message.timestamp,
            messageId: message.id
        };
        
        this.setState('data.healthStatus', healthData);
        
        // Check for health alerts
        if (healthData.overall_status === 'unhealthy') {
            this.addAlert('error', 'System health degraded');
        } else if (healthData.overall_status === 'degraded') {
            this.addAlert('warning', 'System performance degraded');
        }
    }
    
    /**
     * Process heartbeat message
     */
    processHeartbeatMessage(message) {
        // Update connection info if available
        if (message.data.server_time) {
            // Could calculate latency here
        }
    }
    
    /**
     * Add message to message log
     */
    addMessage(message) {
        const messages = [...this.state.data.messages, {
            ...message,
            receivedAt: new Date()
        }];
        
        // Limit message history
        if (messages.length > this.state.data.maxMessages) {
            messages.shift(); // Remove oldest
        }
        
        this.setState('data.messages', messages);
    }
    
    /**
     * Add notification
     */
    addNotification(type, message) {
        const notification = {
            id: Date.now() + Math.random(),
            type,
            message,
            timestamp: new Date()
        };
        
        const notifications = [...this.state.ui.notifications, notification];
        this.setState('ui.notifications', notifications);
        
        // Auto-remove notifications after 5 seconds
        setTimeout(() => {
            this.removeNotification(notification.id);
        }, 5000);
    }
    
    /**
     * Remove notification
     */
    removeNotification(id) {
        const notifications = this.state.ui.notifications.filter(n => n.id !== id);
        this.setState('ui.notifications', notifications);
    }
    
    /**
     * Add alert
     */
    addAlert(type, message) {
        const alert = {
            id: Date.now() + Math.random(),
            type,
            message,
            timestamp: new Date()
        };
        
        const alerts = [...this.state.ui.alerts, alert];
        this.setState('ui.alerts', alerts);
    }
    
    /**
     * Remove alert
     */
    removeAlert(id) {
        const alerts = this.state.ui.alerts.filter(a => a.id !== id);
        this.setState('ui.alerts', alerts);
    }
    
    /**
     * Update subscription settings
     */
    updateSubscription(type, enabled) {
        this.setState(`subscriptions.${type}`, enabled);
        
        // Send subscription update to server if connected
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            const message = {
                type: enabled ? 'subscription' : 'unsubscription',
                data: {
                    types: [type]
                }
            };
            
            this.websocket.send(JSON.stringify(message));
        }
    }
    
    /**
     * Start uptime timer
     */
    startUptimeTimer() {
        this.uptimeTimer = setInterval(() => {
            if (this.state.connection.connectionStartTime) {
                const uptime = Date.now() - this.state.connection.connectionStartTime.getTime();
                this.setState('data.statistics.connectionUptime', uptime);
            }
        }, 1000);
    }
    
    /**
     * Stop uptime timer
     */
    stopUptimeTimer() {
        if (this.uptimeTimer) {
            clearInterval(this.uptimeTimer);
            this.uptimeTimer = null;
        }
    }
    
    /**
     * Get formatted connection uptime
     */
    getFormattedUptime() {
        const uptime = this.state.data.statistics.connectionUptime;
        if (!uptime) return '--';
        
        const seconds = Math.floor(uptime / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);
        
        if (hours > 0) {
            return `${hours}h ${minutes % 60}m`;
        } else if (minutes > 0) {
            return `${minutes}m ${seconds % 60}s`;
        } else {
            return `${seconds}s`;
        }
    }
    
    /**
     * Get task statistics
     */
    getTaskStatistics() {
        const tasks = Array.from(this.state.data.tasks.values());
        
        return {
            total: tasks.length,
            pending: tasks.filter(t => t.status === 'pending').length,
            inProgress: tasks.filter(t => t.status === 'in_progress').length,
            completed: tasks.filter(t => t.status === 'completed').length,
            failed: tasks.filter(t => t.status === 'failed').length
        };
    }
    
    /**
     * Get filtered messages
     */
    getFilteredMessages() {
        let messages = this.state.data.messages;
        
        // Filter by type
        const typeFilter = this.state.ui.filters.messageType;
        if (typeFilter !== 'all') {
            messages = messages.filter(m => m.type === typeFilter);
        }
        
        // Filter by time range
        const timeRange = this.state.ui.filters.timeRange;
        if (timeRange !== 'all') {
            const now = new Date();
            const timeLimit = {
                '1h': 60 * 60 * 1000,
                '6h': 6 * 60 * 60 * 1000,
                '24h': 24 * 60 * 60 * 1000
            }[timeRange];
            
            if (timeLimit) {
                const cutoff = new Date(now.getTime() - timeLimit);
                messages = messages.filter(m => new Date(m.timestamp) > cutoff);
            }
        }
        
        return messages;
    }
    
    /**
     * Initialize event handlers
     */
    initializeEventHandlers() {
        // Handle page visibility changes
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                // Page is hidden - could pause non-critical updates
            } else {
                // Page is visible - resume updates
                if (this.state.connection.status === 'disconnected') {
                    // Try to reconnect when page becomes visible
                    this.connect();
                }
            }
        });
        
        // Handle beforeunload
        window.addEventListener('beforeunload', () => {
            this.disconnect();
        });
    }
    
    /**
     * Export state for debugging
     */
    exportState() {
        return JSON.stringify(this.state, (key, value) => {
            if (value instanceof Map) {
                return Object.fromEntries(value);
            }
            return value;
        }, 2);
    }
    
    /**
     * Reset state to initial values
     */
    reset() {
        this.disconnect();
        
        // Reset to initial state
        this.state = {
            connection: {
                status: 'disconnected',
                url: 'ws://localhost:8765',
                reconnectAttempts: 0,
                maxReconnectAttempts: 5,
                reconnectDelay: 1000,
                lastConnected: null,
                connectionStartTime: null
            },
            subscriptions: {
                telemetry: true,
                task_updates: true,
                system_metrics: true,
                health_status: true,
                heartbeat: true
            },
            data: {
                messages: [],
                maxMessages: 1000,
                telemetry: [],
                tasks: new Map(),
                systemMetrics: {},
                healthStatus: {},
                statistics: {
                    totalMessages: 0,
                    messagesByType: {},
                    connectionUptime: 0,
                    lastMessageTime: null
                }
            },
            ui: {
                selectedTab: 'overview',
                filters: {
                    messageType: 'all',
                    timeRange: '1h'
                },
                notifications: [],
                alerts: []
            }
        };
        
        // Notify all subscribers of reset
        this.notify('*', this.state, null);
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DashboardStateManager;
}

// Also make available globally for browser usage
if (typeof window !== 'undefined') {
    window.DashboardStateManager = DashboardStateManager;
}