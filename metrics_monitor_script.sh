
#!/bin/bash
cd "/Users/anam/temp/0xANATHEMA"

while true; do
    clear
    echo "📊 LABRYS System Metrics - $(date)"
    echo "   " + "="*50
    
    echo "💾 System Resources:"
    echo "   CPU: $(top -l 1 | grep "CPU usage" | awk '{print $3}' | sed 's/%//')"
    echo "   Memory: $(memory_pressure | grep "System-wide memory free percentage" | awk '{print $5}')"
    
    echo ""
    echo "🗲 LABRYS Processes:"
    ps aux | grep -E "(labrys|recursive)" | grep -v grep | wc -l | xargs echo "   Active processes:"
    
    echo ""
    echo "📈 Performance:"
    if [ -f "labrys_self_test_results.json" ]; then
        echo "   Test execution time: $(grep -o '"total_execution_time": [0-9.]*' labrys_self_test_results.json | cut -d' ' -f2)s"
    fi
    
    echo ""
    echo "⏸️  Next update in 30 seconds..."
    sleep 30
done
