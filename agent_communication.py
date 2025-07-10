#!/usr/bin/env python3
"""
Agent Communication Protocols
Atomic Task 49.2: Implement Inter-Agent Communication Protocols

This module implements robust communication protocols for multi-agent coordination,
including message queues, RPC, pub/sub, and reliable messaging mechanisms.
"""

import asyncio
import json
import logging
import pickle
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set, Union
from pathlib import Path
import socket
import threading
from concurrent.futures import ThreadPoolExecutor


class ProtocolType(Enum):
    """Communication protocol types"""
    MESSAGE_QUEUE = "message_queue"
    RPC = "rpc"
    PUB_SUB = "pub_sub"
    DIRECT = "direct"
    BROADCAST = "broadcast"


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class DeliveryMode(Enum):
    """Message delivery modes"""
    FIRE_AND_FORGET = "fire_and_forget"
    ACKNOWLEDGE = "acknowledge"
    GUARANTEED = "guaranteed"
    ORDERED = "ordered"


@dataclass
class MessageEnvelope:
    """Enhanced message envelope with routing and reliability features"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    recipient_id: str = ""
    message_type: str = ""
    payload: Any = None
    priority: MessagePriority = MessagePriority.NORMAL
    delivery_mode: DeliveryMode = DeliveryMode.FIRE_AND_FORGET
    timestamp: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    route: List[str] = field(default_factory=list)
    headers: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['expires_at'] = self.expires_at.isoformat() if self.expires_at else None
        data['priority'] = self.priority.value
        data['delivery_mode'] = self.delivery_mode.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MessageEnvelope':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data['expires_at']:
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        data['priority'] = MessagePriority(data['priority'])
        data['delivery_mode'] = DeliveryMode(data['delivery_mode'])
        return cls(**data)
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        return self.expires_at and datetime.now() > self.expires_at
    
    def should_retry(self) -> bool:
        """Check if message should be retried"""
        return self.retry_count < self.max_retries


@dataclass
class CommunicationStats:
    """Communication statistics tracking"""
    messages_sent: int = 0
    messages_received: int = 0
    messages_failed: int = 0
    messages_retried: int = 0
    average_latency: float = 0.0
    last_activity: datetime = field(default_factory=datetime.now)


class MessageSerializer:
    """Message serialization and deserialization"""
    
    @staticmethod
    def serialize(message: MessageEnvelope) -> bytes:
        """Serialize message to bytes"""
        try:
            # Use JSON for better compatibility
            data = message.to_dict()
            return json.dumps(data).encode('utf-8')
        except Exception:
            # Fallback to pickle for complex objects
            return pickle.dumps(message)
    
    @staticmethod
    def deserialize(data: bytes) -> MessageEnvelope:
        """Deserialize bytes to message"""
        try:
            # Try JSON first
            json_data = json.loads(data.decode('utf-8'))
            return MessageEnvelope.from_dict(json_data)
        except Exception:
            # Fallback to pickle
            return pickle.loads(data)


class CommunicationChannel(ABC):
    """Abstract base class for communication channels"""
    
    def __init__(self, channel_id: str):
        self.channel_id = channel_id
        self.is_active = False
        self.stats = CommunicationStats()
        self.logger = logging.getLogger(f"Channel.{channel_id}")
        self.message_handlers: Dict[str, Callable] = {}
        
    @abstractmethod
    async def send_message(self, message: MessageEnvelope) -> bool:
        """Send a message through this channel"""
        pass
    
    @abstractmethod
    async def receive_message(self) -> Optional[MessageEnvelope]:
        """Receive a message from this channel"""
        pass
    
    @abstractmethod
    async def start(self):
        """Start the communication channel"""
        pass
    
    @abstractmethod
    async def stop(self):
        """Stop the communication channel"""
        pass
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register a message handler"""
        self.message_handlers[message_type] = handler
    
    async def handle_message(self, message: MessageEnvelope):
        """Handle incoming message"""
        handler = self.message_handlers.get(message.message_type)
        if handler:
            try:
                await handler(message)
            except Exception as e:
                self.logger.error(f"Error handling message {message.id}: {e}")


class InMemoryQueue(CommunicationChannel):
    """In-memory message queue for local agent communication"""
    
    def __init__(self, channel_id: str, max_size: int = 1000):
        super().__init__(channel_id)
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.max_size = max_size
        
    async def send_message(self, message: MessageEnvelope) -> bool:
        """Send message to queue"""
        try:
            if message.is_expired():
                self.logger.warning(f"Message {message.id} expired, not sending")
                return False
            
            await self.queue.put(message)
            self.stats.messages_sent += 1
            self.stats.last_activity = datetime.now()
            return True
            
        except asyncio.QueueFull:
            self.logger.error(f"Queue {self.channel_id} is full")
            self.stats.messages_failed += 1
            return False
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            self.stats.messages_failed += 1
            return False
    
    async def receive_message(self) -> Optional[MessageEnvelope]:
        """Receive message from queue"""
        try:
            message = await asyncio.wait_for(self.queue.get(), timeout=1.0)
            self.stats.messages_received += 1
            self.stats.last_activity = datetime.now()
            return message
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            self.logger.error(f"Error receiving message: {e}")
            return None
    
    async def start(self):
        """Start the queue"""
        self.is_active = True
        self.logger.info(f"In-memory queue {self.channel_id} started")
    
    async def stop(self):
        """Stop the queue"""
        self.is_active = False
        self.logger.info(f"In-memory queue {self.channel_id} stopped")


class TCPChannel(CommunicationChannel):
    """TCP-based communication channel for distributed agents"""
    
    def __init__(self, channel_id: str, host: str = "localhost", port: int = 0):
        super().__init__(channel_id)
        self.host = host
        self.port = port
        self.server = None
        self.clients: Dict[str, asyncio.StreamWriter] = {}
        self.pending_messages: asyncio.Queue = asyncio.Queue()
        
    async def send_message(self, message: MessageEnvelope) -> bool:
        """Send message via TCP"""
        try:
            if message.is_expired():
                return False
            
            # Serialize message
            data = MessageSerializer.serialize(message)
            length_prefix = len(data).to_bytes(4, byteorder='big')
            
            # Send to specific recipient or broadcast
            if message.recipient_id in self.clients:
                writer = self.clients[message.recipient_id]
                writer.write(length_prefix + data)
                await writer.drain()
            else:
                # Broadcast to all clients
                for writer in self.clients.values():
                    writer.write(length_prefix + data)
                    await writer.drain()
            
            self.stats.messages_sent += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending TCP message: {e}")
            self.stats.messages_failed += 1
            return False
    
    async def receive_message(self) -> Optional[MessageEnvelope]:
        """Receive message via TCP"""
        try:
            return await asyncio.wait_for(self.pending_messages.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None
    
    async def start(self):
        """Start TCP server"""
        self.server = await asyncio.start_server(
            self._handle_client, self.host, self.port
        )
        self.port = self.server.sockets[0].getsockname()[1]
        self.is_active = True
        self.logger.info(f"TCP channel {self.channel_id} started on {self.host}:{self.port}")
    
    async def stop(self):
        """Stop TCP server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        for writer in self.clients.values():
            writer.close()
            await writer.wait_closed()
        
        self.is_active = False
        self.logger.info(f"TCP channel {self.channel_id} stopped")
    
    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming TCP client"""
        client_addr = writer.get_extra_info('peername')
        client_id = f"{client_addr[0]}:{client_addr[1]}"
        self.clients[client_id] = writer
        
        try:
            while True:
                # Read message length
                length_data = await reader.readexactly(4)
                length = int.from_bytes(length_data, byteorder='big')
                
                # Read message data
                data = await reader.readexactly(length)
                message = MessageSerializer.deserialize(data)
                
                await self.pending_messages.put(message)
                self.stats.messages_received += 1
                
        except asyncio.IncompleteReadError:
            pass  # Client disconnected
        except Exception as e:
            self.logger.error(f"Error handling TCP client {client_id}: {e}")
        finally:
            if client_id in self.clients:
                del self.clients[client_id]
            writer.close()
            await writer.wait_closed()


class PubSubChannel(CommunicationChannel):
    """Publish-Subscribe communication channel"""
    
    def __init__(self, channel_id: str):
        super().__init__(channel_id)
        self.subscribers: Dict[str, Set[Callable]] = {}
        self.message_buffer: Dict[str, List[MessageEnvelope]] = {}
        
    async def send_message(self, message: MessageEnvelope) -> bool:
        """Publish message to topic"""
        try:
            topic = message.headers.get("topic", "default")
            
            # Store message in buffer
            if topic not in self.message_buffer:
                self.message_buffer[topic] = []
            self.message_buffer[topic].append(message)
            
            # Notify subscribers
            if topic in self.subscribers:
                for callback in self.subscribers[topic]:
                    try:
                        await callback(message)
                    except Exception as e:
                        self.logger.error(f"Error notifying subscriber: {e}")
            
            self.stats.messages_sent += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Error publishing message: {e}")
            self.stats.messages_failed += 1
            return False
    
    async def receive_message(self) -> Optional[MessageEnvelope]:
        """Not used in pub/sub - messages are delivered via callbacks"""
        return None
    
    def subscribe(self, topic: str, callback: Callable):
        """Subscribe to a topic"""
        if topic not in self.subscribers:
            self.subscribers[topic] = set()
        self.subscribers[topic].add(callback)
        self.logger.info(f"Subscribed to topic: {topic}")
    
    def unsubscribe(self, topic: str, callback: Callable):
        """Unsubscribe from a topic"""
        if topic in self.subscribers:
            self.subscribers[topic].discard(callback)
            if not self.subscribers[topic]:
                del self.subscribers[topic]
    
    async def start(self):
        """Start pub/sub channel"""
        self.is_active = True
        self.logger.info(f"Pub/Sub channel {self.channel_id} started")
    
    async def stop(self):
        """Stop pub/sub channel"""
        self.is_active = False
        self.subscribers.clear()
        self.message_buffer.clear()
        self.logger.info(f"Pub/Sub channel {self.channel_id} stopped")


class RPCChannel(CommunicationChannel):
    """Remote Procedure Call communication channel"""
    
    def __init__(self, channel_id: str):
        super().__init__(channel_id)
        self.pending_calls: Dict[str, asyncio.Future] = {}
        self.rpc_methods: Dict[str, Callable] = {}
        self.request_queue: asyncio.Queue = asyncio.Queue()
        
    async def send_message(self, message: MessageEnvelope) -> bool:
        """Send RPC message"""
        try:
            if message.message_type == "rpc_request":
                await self.request_queue.put(message)
            elif message.message_type == "rpc_response":
                # Handle response
                if message.correlation_id in self.pending_calls:
                    future = self.pending_calls[message.correlation_id]
                    if not future.done():
                        future.set_result(message.payload)
                    del self.pending_calls[message.correlation_id]
            
            self.stats.messages_sent += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending RPC message: {e}")
            self.stats.messages_failed += 1
            return False
    
    async def receive_message(self) -> Optional[MessageEnvelope]:
        """Receive RPC request"""
        try:
            return await asyncio.wait_for(self.request_queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None
    
    async def call(self, method: str, args: List[Any] = None, kwargs: Dict[str, Any] = None, timeout: float = 30.0) -> Any:
        """Make RPC call"""
        call_id = str(uuid.uuid4())
        
        request = MessageEnvelope(
            id=call_id,
            message_type="rpc_request",
            payload={
                "method": method,
                "args": args or [],
                "kwargs": kwargs or {}
            },
            correlation_id=call_id
        )
        
        # Create future for response
        future = asyncio.Future()
        self.pending_calls[call_id] = future
        
        # Send request
        await self.send_message(request)
        
        try:
            # Wait for response
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            if call_id in self.pending_calls:
                del self.pending_calls[call_id]
            raise RuntimeError(f"RPC call to {method} timed out")
    
    def register_method(self, name: str, method: Callable):
        """Register RPC method"""
        self.rpc_methods[name] = method
        self.logger.info(f"Registered RPC method: {name}")
    
    async def start(self):
        """Start RPC channel"""
        self.is_active = True
        # Start request processing loop
        asyncio.create_task(self._process_requests())
        self.logger.info(f"RPC channel {self.channel_id} started")
    
    async def stop(self):
        """Stop RPC channel"""
        self.is_active = False
        
        # Cancel pending calls
        for future in self.pending_calls.values():
            if not future.done():
                future.cancel()
        self.pending_calls.clear()
        
        self.logger.info(f"RPC channel {self.channel_id} stopped")
    
    async def _process_requests(self):
        """Process incoming RPC requests"""
        while self.is_active:
            try:
                request = await self.receive_message()
                if request and request.message_type == "rpc_request":
                    asyncio.create_task(self._handle_rpc_request(request))
            except Exception as e:
                self.logger.error(f"Error processing RPC requests: {e}")
    
    async def _handle_rpc_request(self, request: MessageEnvelope):
        """Handle individual RPC request"""
        try:
            payload = request.payload
            method_name = payload["method"]
            args = payload["args"]
            kwargs = payload["kwargs"]
            
            if method_name in self.rpc_methods:
                method = self.rpc_methods[method_name]
                result = await method(*args, **kwargs) if asyncio.iscoroutinefunction(method) else method(*args, **kwargs)
                
                # Send response
                response = MessageEnvelope(
                    message_type="rpc_response",
                    payload=result,
                    correlation_id=request.correlation_id,
                    recipient_id=request.sender_id
                )
                await self.send_message(response)
            else:
                # Method not found
                error_response = MessageEnvelope(
                    message_type="rpc_response",
                    payload={"error": f"Method {method_name} not found"},
                    correlation_id=request.correlation_id,
                    recipient_id=request.sender_id
                )
                await self.send_message(error_response)
                
        except Exception as e:
            # Send error response
            error_response = MessageEnvelope(
                message_type="rpc_response",
                payload={"error": str(e)},
                correlation_id=request.correlation_id,
                recipient_id=request.sender_id
            )
            await self.send_message(error_response)


class ReliableMessaging:
    """Reliable messaging with acknowledgments and retries"""
    
    def __init__(self, channel: CommunicationChannel):
        self.channel = channel
        self.pending_acks: Dict[str, MessageEnvelope] = {}
        self.ack_timeout = 30.0  # seconds
        self.retry_intervals = [1, 2, 4, 8]  # exponential backoff
        self.logger = logging.getLogger("ReliableMessaging")
    
    async def send_reliable(self, message: MessageEnvelope) -> bool:
        """Send message with reliability guarantees"""
        if message.delivery_mode == DeliveryMode.FIRE_AND_FORGET:
            return await self.channel.send_message(message)
        
        # Store for potential retry
        self.pending_acks[message.id] = message
        
        # Send with acknowledgment request
        message.headers["require_ack"] = True
        success = await self.channel.send_message(message)
        
        if success and message.delivery_mode == DeliveryMode.ACKNOWLEDGE:
            # Wait for acknowledgment
            await self._wait_for_ack(message.id)
        
        return success
    
    async def _wait_for_ack(self, message_id: str):
        """Wait for message acknowledgment"""
        start_time = time.time()
        
        while time.time() - start_time < self.ack_timeout:
            if message_id not in self.pending_acks:
                return  # Acknowledged
            await asyncio.sleep(0.1)
        
        # Timeout - retry if possible
        if message_id in self.pending_acks:
            message = self.pending_acks[message_id]
            if message.should_retry():
                message.retry_count += 1
                retry_interval = self.retry_intervals[min(message.retry_count - 1, len(self.retry_intervals) - 1)]
                await asyncio.sleep(retry_interval)
                await self.send_reliable(message)
            else:
                del self.pending_acks[message_id]
                self.logger.error(f"Message {message_id} failed after max retries")
    
    def acknowledge(self, message_id: str):
        """Acknowledge message receipt"""
        if message_id in self.pending_acks:
            del self.pending_acks[message_id]


class MessageRouter:
    """Message routing and load balancing"""
    
    def __init__(self):
        self.channels: Dict[str, CommunicationChannel] = {}
        self.routing_table: Dict[str, str] = {}  # agent_id -> channel_id
        self.load_balancer_strategies: Dict[str, Callable] = {
            "round_robin": self._round_robin,
            "least_loaded": self._least_loaded,
            "random": self._random_select
        }
        self.current_strategy = "round_robin"
        self.round_robin_index = 0
        self.logger = logging.getLogger("MessageRouter")
    
    def add_channel(self, channel: CommunicationChannel):
        """Add communication channel"""
        self.channels[channel.channel_id] = channel
        self.logger.info(f"Added channel: {channel.channel_id}")
    
    def register_agent(self, agent_id: str, channel_id: str):
        """Register agent with specific channel"""
        self.routing_table[agent_id] = channel_id
        self.logger.info(f"Registered agent {agent_id} on channel {channel_id}")
    
    async def route_message(self, message: MessageEnvelope) -> bool:
        """Route message to appropriate channel"""
        # Direct routing to specific agent
        if message.recipient_id in self.routing_table:
            channel_id = self.routing_table[message.recipient_id]
            if channel_id in self.channels:
                return await self.channels[channel_id].send_message(message)
        
        # Broadcast or load balance
        if message.recipient_id == "broadcast":
            return await self._broadcast_message(message)
        else:
            return await self._load_balance_message(message)
    
    async def _broadcast_message(self, message: MessageEnvelope) -> bool:
        """Broadcast message to all channels"""
        results = []
        for channel in self.channels.values():
            result = await channel.send_message(message)
            results.append(result)
        return all(results)
    
    async def _load_balance_message(self, message: MessageEnvelope) -> bool:
        """Load balance message across channels"""
        strategy = self.load_balancer_strategies[self.current_strategy]
        channel = strategy()
        if channel:
            return await channel.send_message(message)
        return False
    
    def _round_robin(self) -> Optional[CommunicationChannel]:
        """Round robin channel selection"""
        if not self.channels:
            return None
        
        channel_list = list(self.channels.values())
        channel = channel_list[self.round_robin_index % len(channel_list)]
        self.round_robin_index += 1
        return channel
    
    def _least_loaded(self) -> Optional[CommunicationChannel]:
        """Select least loaded channel"""
        if not self.channels:
            return None
        
        return min(self.channels.values(), key=lambda c: c.stats.messages_sent - c.stats.messages_received)
    
    def _random_select(self) -> Optional[CommunicationChannel]:
        """Random channel selection"""
        import random
        if not self.channels:
            return None
        return random.choice(list(self.channels.values()))


class CommunicationManager:
    """Central manager for all communication protocols"""
    
    def __init__(self):
        self.channels: Dict[str, CommunicationChannel] = {}
        self.router = MessageRouter()
        self.reliable_messaging = {}
        self.is_running = False
        self.logger = logging.getLogger("CommunicationManager")
    
    def create_channel(self, channel_type: ProtocolType, channel_id: str, **kwargs) -> CommunicationChannel:
        """Create communication channel"""
        if channel_type == ProtocolType.MESSAGE_QUEUE:
            channel = InMemoryQueue(channel_id, **kwargs)
        elif channel_type == ProtocolType.RPC:
            channel = RPCChannel(channel_id)
        elif channel_type == ProtocolType.PUB_SUB:
            channel = PubSubChannel(channel_id)
        elif channel_type == ProtocolType.DIRECT:
            channel = TCPChannel(channel_id, **kwargs)
        else:
            raise ValueError(f"Unsupported channel type: {channel_type}")
        
        self.channels[channel_id] = channel
        self.router.add_channel(channel)
        self.reliable_messaging[channel_id] = ReliableMessaging(channel)
        
        self.logger.info(f"Created {channel_type.value} channel: {channel_id}")
        return channel
    
    async def start_all(self):
        """Start all communication channels"""
        self.is_running = True
        
        for channel in self.channels.values():
            await channel.start()
        
        self.logger.info("All communication channels started")
    
    async def stop_all(self):
        """Stop all communication channels"""
        self.is_running = False
        
        for channel in self.channels.values():
            await channel.stop()
        
        self.logger.info("All communication channels stopped")
    
    async def send_message(self, message: MessageEnvelope, reliable: bool = False) -> bool:
        """Send message through appropriate channel"""
        if reliable:
            # Use reliable messaging
            channel_id = self.router.routing_table.get(message.recipient_id)
            if channel_id in self.reliable_messaging:
                return await self.reliable_messaging[channel_id].send_reliable(message)
        
        return await self.router.route_message(message)
    
    def register_agent(self, agent_id: str, channel_id: str):
        """Register agent with communication manager"""
        self.router.register_agent(agent_id, channel_id)
    
    def get_stats(self) -> Dict[str, CommunicationStats]:
        """Get communication statistics"""
        return {cid: channel.stats for cid, channel in self.channels.items()}


# Export key classes
__all__ = [
    "ProtocolType", "MessagePriority", "DeliveryMode", "MessageEnvelope",
    "CommunicationStats", "MessageSerializer", "CommunicationChannel",
    "InMemoryQueue", "TCPChannel", "PubSubChannel", "RPCChannel",
    "ReliableMessaging", "MessageRouter", "CommunicationManager"
]


if __name__ == "__main__":
    # Demo usage
    async def demo():
        logging.basicConfig(level=logging.INFO)
        
        # Create communication manager
        comm_manager = CommunicationManager()
        
        # Create channels
        queue_channel = comm_manager.create_channel(ProtocolType.MESSAGE_QUEUE, "main_queue")
        rpc_channel = comm_manager.create_channel(ProtocolType.RPC, "rpc_service")
        pubsub_channel = comm_manager.create_channel(ProtocolType.PUB_SUB, "events")
        
        # Start all channels
        await comm_manager.start_all()
        
        # Register some agents
        comm_manager.register_agent("agent1", "main_queue")
        comm_manager.register_agent("agent2", "main_queue")
        
        # Send test messages
        test_message = MessageEnvelope(
            sender_id="test_sender",
            recipient_id="agent1",
            message_type="test_task",
            payload={"task": "process_data", "data": [1, 2, 3]},
            priority=MessagePriority.HIGH
        )
        
        success = await comm_manager.send_message(test_message)
        print(f"Message sent: {success}")
        
        # Get statistics
        stats = comm_manager.get_stats()
        print("Communication Stats:")
        for channel_id, stat in stats.items():
            print(f"  {channel_id}: {stat.messages_sent} sent, {stat.messages_received} received")
        
        # Stop all channels
        await comm_manager.stop_all()
    
    # Run demo
    asyncio.run(demo())