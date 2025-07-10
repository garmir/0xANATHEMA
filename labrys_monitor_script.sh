
#!/bin/bash
cd "/Users/anam/temp/0xANATHEMA"
source venv/bin/activate

echo "🗲 LABRYS Process Monitor"
echo "   Real-time monitoring of 5 processes"
echo "   " + "="*50

while true; do
    clear
    echo "🗲 LABRYS Process Monitor - $(date)"
    echo "   " + "="*50
    
    # Health check
    python3 check_labrys_health.py
    
    echo ""
    echo "📊 Process Guardian Status:"
    if ps aux | grep -q "labrys_process_guardian"; then
        echo "   🛡️  Guardian: ACTIVE"
    else
        echo "   ⚠️  Guardian: NOT RUNNING"
    fi
    
    echo ""
    echo "📋 PID Scenarios:"
    if [ -d ".labrys/pid_scenarios" ]; then
        for scenario in .labrys/pid_scenarios/*/; do
            if [ -f "$scenario/current_status.json" ]; then
                pid=$(basename "$scenario" | sed 's/pid_//')
                health=$(python3 -c "import json; data=json.load(open('$scenario/current_status.json')); print(f'Health: {data["health_score"]:.1f} - {data["process_stats"]["status"]}')")
                echo "   PID $pid: $health"
            fi
        done
    fi
    
    echo ""
    echo "🔄 Refreshing in 5 seconds... (Ctrl+C to exit)"
    sleep 5
done
