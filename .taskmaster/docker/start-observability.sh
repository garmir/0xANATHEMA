#!/bin/bash
# Task-Master Observability Stack Startup Script

echo "🚀 Starting Task-Master Observability Stack..."

# Start the stack
docker-compose up -d

# Wait for services
echo "⏳ Waiting for services to be ready..."
sleep 15

# Check service health
echo "🏥 Checking service health..."

# Check Jaeger
if curl -s http://localhost:16686/api/services > /dev/null; then
    echo "✅ Jaeger is ready: http://localhost:16686"
else
    echo "❌ Jaeger is not responding"
fi

# Check Prometheus
if curl -s http://localhost:9090/-/ready > /dev/null; then
    echo "✅ Prometheus is ready: http://localhost:9090"
else
    echo "❌ Prometheus is not responding"
fi

# Check Grafana
if curl -s http://localhost:3000/api/health > /dev/null; then
    echo "✅ Grafana is ready: http://localhost:3000 (admin/admin)"
else
    echo "❌ Grafana is not responding"
fi

# Check OTEL Collector
if curl -s http://localhost:13133/health > /dev/null; then
    echo "✅ OTEL Collector is ready: http://localhost:13133/health"
else
    echo "❌ OTEL Collector is not responding"
fi

echo ""
echo "🎯 Observability Stack Status:"
echo "  📊 Jaeger UI: http://localhost:16686"
echo "  📈 Prometheus: http://localhost:9090"
echo "  📉 Grafana: http://localhost:3000"
echo "  🔍 OTEL Health: http://localhost:13133/health"
echo ""
echo "💡 Next steps:"
echo "  1. Run your Task-Master application with OpenTelemetry enabled"
echo "  2. View traces in Jaeger"
echo "  3. Check metrics in Prometheus"
echo "  4. Create dashboards in Grafana"
