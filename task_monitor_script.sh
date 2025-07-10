
#!/bin/bash
cd "/Users/anam/temp/0xANATHEMA"

while true; do
    clear
    echo "📋 LABRYS Task Completion Monitor - $(date)"
    echo "   " + "="*50
    
    echo "🎯 Active Scenarios:"
    if [ -d ".labrys/pid_scenarios" ]; then
        for scenario in .labrys/pid_scenarios/*/; do
            if [ -f "$scenario/current_status.json" ]; then
                pid=$(basename "$scenario" | sed 's/pid_//')
                echo "   PID $pid: Monitoring active"
            fi
        done
    else
        echo "   No active scenarios"
    fi
    
    echo ""
    echo "🏁 Completed Tasks:"
    if [ -f "labrys_self_test_results.json" ]; then
        echo "   Self-test: COMPLETED"
    fi
    if [ -f "recursive_improvement_results.json" ]; then
        echo "   Recursive improvement: COMPLETED"
    fi
    
    echo ""
    echo "⏸️  Next update in 20 seconds..."
    sleep 20
done
