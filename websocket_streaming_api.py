#!/usr/bin/env python3
"""
WebSocket Streaming API for Task-Master Dashboard
Implements atomic task: Implement backend WebSocket streaming API

Based on research-driven breakdown:
- Create WebSocket server for real-time telemetry streaming
- Implement data filtering and subscription management
- Add connection pooling and message broadcasting
- Ensure scalable real-time data delivery to dashboards

This module provides a WebSocket API that streams telemetry data,
task updates, and system metrics to frontend dashboards in real-time.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Any, List, Set, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import weakref

# WebSocket implementation using asyncio
try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logging.warning("websockets library not available - using simulation mode")

# Import our correlation system
from melt_correlation_system import get_correlation_manager, CorrelatedOperation

class MessageType(Enum):
    """Types of WebSocket messages"""
    TELEMETRY = "telemetry"
    TASK_UPDATE = "task_update"
    SYSTEM_METRIC = "system_metric"
    HEALTH_STATUS = "health_status"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    SUBSCRIPTION = "subscription"
    UNSUBSCRIPTION = "unsubscription"

class SubscriptionType(Enum):
    """Types of data subscriptions"""
    ALL = "all"
    TELEMETRY_ONLY = "telemetry_only"
    TASKS_ONLY = "tasks_only"
    METRICS_ONLY = "metrics_only"
    HEALTH_ONLY = "health_only"

@dataclass
class WebSocketMessage:
    """Standard WebSocket message format"""
    id: str
    type: MessageType
    timestamp: str
    data: Dict[str, Any]
    correlation_id: Optional[str] = None
    
    def to_json(self) -> str:
        """Convert message to JSON string"""
        return json.dumps({
            "id": self.id,
            "type": self.type.value,
            "timestamp": self.timestamp,
            "data": self.data,
            "correlation_id": self.correlation_id
        })
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WebSocketMessage':
        """Create message from dictionary"""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=MessageType(data.get("type", "telemetry")),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            data=data.get("data", {}),
            correlation_id=data.get("correlation_id")
        )

@dataclass
class ClientConnection:
    """WebSocket client connection info"""
    id: str
    websocket: Any  # WebSocket connection
    subscriptions: Set[SubscriptionType]
    connected_at: datetime
    last_heartbeat: datetime
    client_info: Dict[str, Any]
    message_count: int = 0
    
    def is_alive(self) -> bool:
        """Check if connection is still alive"""
        if not WEBSOCKETS_AVAILABLE:
            return True  # Simulate alive in demo mode
        
        # Check if websocket is still open
        if hasattr(self.websocket, 'closed'):
            return not self.websocket.closed
        return True
    
    def wants_message_type(self, message_type: MessageType) -> bool:
        """Check if client is subscribed to message type"""
        if SubscriptionType.ALL in self.subscriptions:
            return True
        
        type_mapping = {
            MessageType.TELEMETRY: SubscriptionType.TELEMETRY_ONLY,
            MessageType.TASK_UPDATE: SubscriptionType.TASKS_ONLY,
            MessageType.SYSTEM_METRIC: SubscriptionType.METRICS_ONLY,
            MessageType.HEALTH_STATUS: SubscriptionType.HEALTH_ONLY,
        }
        
        required_subscription = type_mapping.get(message_type)
        return required_subscription in self.subscriptions if required_subscription else True

class WebSocketStreamingServer:
    """
    WebSocket server for streaming telemetry data to dashboards
    """
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Dict[str, ClientConnection] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.correlation_manager = get_correlation_manager()
        self.server = None
        self.broadcaster_task = None
        self.heartbeat_task = None
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "messages_queued": 0,
            "started_at": None
        }
        
        logging.info(f"WebSocket streaming server initialized on {host}:{port}")
    
    async def start_server(self):
        """Start the WebSocket server"""
        if not WEBSOCKETS_AVAILABLE:
            logging.warning("WebSocket server starting in simulation mode")
            self.stats["started_at"] = datetime.now().isoformat()
            
            # Start background tasks in simulation mode
            self.broadcaster_task = asyncio.create_task(self._message_broadcaster())
            self.heartbeat_task = asyncio.create_task(self._heartbeat_manager())
            
            # Simulate some client connections
            await self._simulate_clients()
            return
        
        try:
            self.server = await websockets.serve(
                self._handle_client_connection,
                self.host,
                self.port,
                ping_interval=30,
                ping_timeout=10
            )
            
            self.stats["started_at"] = datetime.now().isoformat()
            
            # Start background tasks
            self.broadcaster_task = asyncio.create_task(self._message_broadcaster())
            self.heartbeat_task = asyncio.create_task(self._heartbeat_manager())
            
            logging.info(f"✅ WebSocket server started on ws://{self.host}:{self.port}")
            
        except Exception as e:
            logging.error(f"Failed to start WebSocket server: {e}")
            raise
    
    async def stop_server(self):
        """Stop the WebSocket server"""
        if self.broadcaster_task:
            self.broadcaster_task.cancel()
        
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        
        if self.server and WEBSOCKETS_AVAILABLE:
            self.server.close()
            await self.server.wait_closed()
        
        # Close all client connections
        for client in list(self.clients.values()):
            if WEBSOCKETS_AVAILABLE and hasattr(client.websocket, 'close'):
                await client.websocket.close()
        
        self.clients.clear()
        logging.info("WebSocket server stopped")
    
    async def _handle_client_connection(self, websocket, path):
        """Handle new WebSocket client connection"""
        client_id = str(uuid.uuid4())
        
        try:
            # Get client info from headers or query params
            client_info = {
                "user_agent": websocket.request_headers.get("User-Agent", "Unknown"),
                "origin": websocket.request_headers.get("Origin", "Unknown"),
                "remote_address": websocket.remote_address[0] if websocket.remote_address else "Unknown"
            }
            
            # Create client connection
            client = ClientConnection(
                id=client_id,
                websocket=websocket,
                subscriptions={SubscriptionType.ALL},  # Default to all subscriptions
                connected_at=datetime.now(),
                last_heartbeat=datetime.now(),
                client_info=client_info
            )
            
            self.clients[client_id] = client
            self.stats["total_connections"] += 1
            self.stats["active_connections"] = len(self.clients)
            
            logging.info(f"✅ Client {client_id} connected from {client_info['remote_address']}")
            
            # Send welcome message
            welcome_message = WebSocketMessage(
                id=str(uuid.uuid4()),
                type=MessageType.HEARTBEAT,
                timestamp=datetime.now().isoformat(),
                data={
                    "status": "connected",
                    "client_id": client_id,
                    "server_info": {
                        "version": "1.0.0",
                        "features": ["telemetry", "tasks", "metrics", "health"]
                    }
                }
            )
            
            await websocket.send(welcome_message.to_json())
            
            # Handle client messages
            async for raw_message in websocket:
                await self._handle_client_message(client_id, raw_message)
                
        except websockets.exceptions.ConnectionClosed:
            logging.info(f"Client {client_id} disconnected")
        except Exception as e:
            logging.error(f"Error handling client {client_id}: {e}")
        finally:
            # Clean up client connection
            if client_id in self.clients:
                del self.clients[client_id]
                self.stats["active_connections"] = len(self.clients)
    
    async def _handle_client_message(self, client_id: str, raw_message: str):
        """Handle message from WebSocket client"""
        try:
            message_data = json.loads(raw_message)
            message = WebSocketMessage.from_dict(message_data)
            
            client = self.clients.get(client_id)
            if not client:
                return
            
            client.last_heartbeat = datetime.now()
            client.message_count += 1
            
            # Handle different message types
            if message.type == MessageType.SUBSCRIPTION:
                await self._handle_subscription(client_id, message.data)
            elif message.type == MessageType.UNSUBSCRIPTION:
                await self._handle_unsubscription(client_id, message.data)
            elif message.type == MessageType.HEARTBEAT:
                # Update heartbeat timestamp
                pass
            else:
                logging.warning(f"Unknown message type from client {client_id}: {message.type}")
                
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON from client {client_id}")
        except Exception as e:
            logging.error(f"Error processing message from client {client_id}: {e}")
    
    async def _handle_subscription(self, client_id: str, data: Dict[str, Any]):
        """Handle subscription request from client"""
        client = self.clients.get(client_id)
        if not client:
            return
        
        subscription_types = data.get("types", [])
        
        for sub_type in subscription_types:
            try:
                subscription = SubscriptionType(sub_type)
                client.subscriptions.add(subscription)
            except ValueError:
                logging.warning(f"Invalid subscription type: {sub_type}")
        
        logging.info(f"Client {client_id} subscribed to: {[s.value for s in client.subscriptions]}")
    
    async def _handle_unsubscription(self, client_id: str, data: Dict[str, Any]):
        """Handle unsubscription request from client"""
        client = self.clients.get(client_id)
        if not client:
            return
        
        subscription_types = data.get("types", [])
        
        for sub_type in subscription_types:
            try:
                subscription = SubscriptionType(sub_type)
                client.subscriptions.discard(subscription)
            except ValueError:
                logging.warning(f"Invalid subscription type: {sub_type}")
        
        logging.info(f"Client {client_id} unsubscribed from: {subscription_types}")
    
    async def _message_broadcaster(self):
        """Background task to broadcast messages to clients"""
        while True:
            try:
                # Get message from queue with timeout
                try:
                    message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Broadcast to all interested clients
                await self._broadcast_message(message)
                self.stats["messages_sent"] += len(self.clients)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in message broadcaster: {e}")
                await asyncio.sleep(1)
    
    async def _broadcast_message(self, message: WebSocketMessage):
        """Broadcast message to all interested clients"""
        if not self.clients:
            return
        
        # Find clients interested in this message type
        interested_clients = [
            client for client in self.clients.values()
            if client.is_alive() and client.wants_message_type(message.type)
        ]
        
        if not interested_clients:
            return
        
        # Send message to all interested clients
        message_json = message.to_json()
        
        for client in interested_clients:
            try:
                if WEBSOCKETS_AVAILABLE and hasattr(client.websocket, 'send'):
                    await client.websocket.send(message_json)
                else:
                    # Simulate sending in demo mode
                    logging.debug(f"📤 Simulated send to client {client.id}: {message.type.value}")
                    
            except Exception as e:
                logging.error(f"Failed to send message to client {client.id}: {e}")
                # Remove failed client
                if client.id in self.clients:
                    del self.clients[client.id]
                    self.stats["active_connections"] = len(self.clients)
    
    async def _heartbeat_manager(self):
        """Background task to manage client heartbeats"""
        while True:
            try:
                current_time = datetime.now()
                stale_threshold = timedelta(minutes=2)
                
                # Find stale clients
                stale_clients = []
                for client_id, client in self.clients.items():
                    if current_time - client.last_heartbeat > stale_threshold:
                        stale_clients.append(client_id)
                
                # Remove stale clients
                for client_id in stale_clients:
                    logging.info(f"Removing stale client: {client_id}")
                    client = self.clients.get(client_id)
                    if client and WEBSOCKETS_AVAILABLE and hasattr(client.websocket, 'close'):
                        try:
                            await client.websocket.close()
                        except:
                            pass
                    
                    if client_id in self.clients:
                        del self.clients[client_id]
                
                self.stats["active_connections"] = len(self.clients)
                
                # Send heartbeat to remaining clients
                if self.clients:
                    heartbeat_message = WebSocketMessage(
                        id=str(uuid.uuid4()),
                        type=MessageType.HEARTBEAT,
                        timestamp=current_time.isoformat(),
                        data={
                            "server_time": current_time.isoformat(),
                            "active_clients": len(self.clients)
                        }
                    )
                    
                    await self._broadcast_message(heartbeat_message)
                
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in heartbeat manager: {e}")
                await asyncio.sleep(30)
    
    async def _simulate_clients(self):
        """Simulate client connections for demo purposes"""
        # Simulate 3 dashboard clients
        for i in range(3):
            client_id = f"demo_client_{i}"
            client = ClientConnection(
                id=client_id,
                websocket=None,  # No real websocket in demo
                subscriptions={SubscriptionType.ALL},
                connected_at=datetime.now(),
                last_heartbeat=datetime.now(),
                client_info={
                    "user_agent": f"Demo Dashboard {i}",
                    "origin": "http://localhost:3000",
                    "remote_address": f"127.0.0.{i+1}"
                }
            )
            
            self.clients[client_id] = client
            self.stats["total_connections"] += 1
        
        self.stats["active_connections"] = len(self.clients)
        logging.info(f"🎭 Simulated {len(self.clients)} dashboard clients")
    
    async def send_telemetry_data(self, telemetry_data: Dict[str, Any]):
        """Send telemetry data to connected clients"""
        message = WebSocketMessage(
            id=str(uuid.uuid4()),
            type=MessageType.TELEMETRY,
            timestamp=datetime.now().isoformat(),
            data=telemetry_data,
            correlation_id=telemetry_data.get("trace_id")
        )
        
        await self.message_queue.put(message)
        self.stats["messages_queued"] += 1
    
    async def send_task_update(self, task_data: Dict[str, Any]):
        """Send task update to connected clients"""
        message = WebSocketMessage(
            id=str(uuid.uuid4()),
            type=MessageType.TASK_UPDATE,
            timestamp=datetime.now().isoformat(),
            data=task_data
        )
        
        await self.message_queue.put(message)
        self.stats["messages_queued"] += 1
    
    async def send_system_metric(self, metric_data: Dict[str, Any]):
        """Send system metric to connected clients"""
        message = WebSocketMessage(
            id=str(uuid.uuid4()),
            type=MessageType.SYSTEM_METRIC,
            timestamp=datetime.now().isoformat(),
            data=metric_data
        )
        
        await self.message_queue.put(message)
        self.stats["messages_queued"] += 1
    
    async def send_health_status(self, health_data: Dict[str, Any]):
        """Send health status to connected clients"""
        message = WebSocketMessage(
            id=str(uuid.uuid4()),
            type=MessageType.HEALTH_STATUS,
            timestamp=datetime.now().isoformat(),
            data=health_data
        )
        
        await self.message_queue.put(message)
        self.stats["messages_queued"] += 1
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get server statistics"""
        uptime = None
        if self.stats["started_at"]:
            start_time = datetime.fromisoformat(self.stats["started_at"])
            uptime = (datetime.now() - start_time).total_seconds()
        
        return {
            **self.stats,
            "uptime_seconds": uptime,
            "queue_size": self.message_queue.qsize(),
            "client_details": [
                {
                    "id": client.id,
                    "connected_at": client.connected_at.isoformat(),
                    "last_heartbeat": client.last_heartbeat.isoformat(),
                    "subscriptions": [s.value for s in client.subscriptions],
                    "message_count": client.message_count,
                    "client_info": client.client_info
                }
                for client in self.clients.values()
            ]
        }

class TelemetryDataGenerator:
    """
    Generates sample telemetry data for demonstration
    """
    
    def __init__(self):
        self.correlation_manager = get_correlation_manager()
        self.task_counter = 0
        self.metric_counter = 0
    
    async def generate_sample_data(self, server: WebSocketStreamingServer, duration: int = 30):
        """Generate sample telemetry data for specified duration"""
        end_time = time.time() + duration
        
        while time.time() < end_time:
            # Generate different types of data
            await self._generate_telemetry_data(server)
            await asyncio.sleep(2)
            
            await self._generate_task_update(server)
            await asyncio.sleep(1)
            
            await self._generate_system_metric(server)
            await asyncio.sleep(1)
            
            await self._generate_health_status(server)
            await asyncio.sleep(3)
    
    async def _generate_telemetry_data(self, server: WebSocketStreamingServer):
        """Generate sample telemetry data"""
        with CorrelatedOperation(self.correlation_manager, "sample_telemetry_generation") as context:
            telemetry_data = self.correlation_manager.correlate_metric(
                "sample.telemetry.metric",
                float(time.time() % 100),
                {
                    "component": "telemetry_generator",
                    "type": "demo_data",
                    "sequence": self.metric_counter
                }
            )
            
            await server.send_telemetry_data(telemetry_data)
            self.metric_counter += 1
    
    async def _generate_task_update(self, server: WebSocketStreamingServer):
        """Generate sample task update"""
        self.task_counter += 1
        
        task_data = {
            "task_id": f"task_{self.task_counter:03d}",
            "status": "in_progress" if self.task_counter % 3 != 0 else "completed",
            "progress": min(100, (self.task_counter * 7) % 100),
            "type": "atomic_task",
            "description": f"Sample task {self.task_counter}",
            "updated_at": datetime.now().isoformat()
        }
        
        await server.send_task_update(task_data)
    
    async def _generate_system_metric(self, server: WebSocketStreamingServer):
        """Generate sample system metric"""
        import random
        
        metric_data = {
            "metric_name": "system.cpu.usage",
            "value": random.uniform(10, 90),
            "unit": "percent",
            "host": "task-master-host",
            "timestamp": datetime.now().isoformat()
        }
        
        await server.send_system_metric(metric_data)
    
    async def _generate_health_status(self, server: WebSocketStreamingServer):
        """Generate sample health status"""
        import random
        
        services = ["jaeger", "prometheus", "grafana", "otel-collector"]
        
        health_data = {
            "overall_status": "healthy",
            "services": {
                service: {
                    "status": "healthy" if random.random() > 0.1 else "degraded",
                    "response_time_ms": random.uniform(10, 200),
                    "last_check": datetime.now().isoformat()
                }
                for service in services
            }
        }
        
        await server.send_health_status(health_data)

async def demonstrate_websocket_streaming():
    """Demonstrate WebSocket streaming functionality"""
    print("🌐 WebSocket Streaming API Demo")
    print("=" * 60)
    
    # Initialize server
    server = WebSocketStreamingServer()
    data_generator = TelemetryDataGenerator()
    
    try:
        # Start server
        print("🚀 Starting WebSocket server...")
        await server.start_server()
        
        print(f"📡 Server running on ws://{server.host}:{server.port}")
        print(f"👥 Connected clients: {server.stats['active_connections']}")
        
        # Generate sample data
        print("\n📊 Generating sample telemetry data...")
        await data_generator.generate_sample_data(server, duration=15)
        
        # Show server statistics
        stats = server.get_server_stats()
        print(f"\n📈 Server Statistics:")
        print(f"  Total Connections: {stats['total_connections']}")
        print(f"  Active Connections: {stats['active_connections']}")
        print(f"  Messages Sent: {stats['messages_sent']}")
        print(f"  Messages Queued: {stats['messages_queued']}")
        print(f"  Queue Size: {stats['queue_size']}")
        print(f"  Uptime: {stats['uptime_seconds']:.1f} seconds")
        
        print(f"\n👥 Client Details:")
        for client in stats['client_details']:
            print(f"  📱 {client['id'][:8]}...")
            print(f"    Subscriptions: {', '.join(client['subscriptions'])}")
            print(f"    Messages: {client['message_count']}")
            print(f"    Source: {client['client_info']['remote_address']}")
        
        # Save demonstration results
        demo_results = {
            "demonstration_completed_at": datetime.now().isoformat(),
            "server_configuration": {
                "host": server.host,
                "port": server.port,
                "websockets_available": WEBSOCKETS_AVAILABLE
            },
            "server_statistics": stats,
            "features_demonstrated": [
                "Real-time WebSocket streaming",
                "Multiple client connection management",
                "Message type filtering and subscriptions",
                "Telemetry data broadcasting",
                "Task update streaming",
                "System metrics streaming",
                "Health status monitoring",
                "Connection lifecycle management"
            ]
        }
        
        # Save results
        import os
        os.makedirs(".taskmaster/reports", exist_ok=True)
        
        with open(".taskmaster/reports/websocket-streaming-demo.json", 'w') as f:
            json.dump(demo_results, f, indent=2)
        
        print(f"\n📄 Demo results saved to: .taskmaster/reports/websocket-streaming-demo.json")
        
    finally:
        print("\n🛑 Stopping WebSocket server...")
        await server.stop_server()
    
    print("\n✅ WebSocket streaming demonstration completed!")
    print("🎯 Features implemented:")
    print("  • Real-time WebSocket server")
    print("  • Multi-client connection management")
    print("  • Message type filtering and subscriptions")
    print("  • Telemetry data streaming")
    print("  • Connection lifecycle management")
    print("  • Heartbeat and health monitoring")

if __name__ == "__main__":
    try:
        asyncio.run(demonstrate_websocket_streaming())
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Error during WebSocket demonstration: {e}")
        import traceback
        traceback.print_exc()