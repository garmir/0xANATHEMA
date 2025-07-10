
#!/bin/bash
cd "/Users/anam/temp/0xANATHEMA"

while true; do
    clear
    echo "🛡️  LABRYS Process Guardian Monitor - $(date)"
    echo "   " + "="*50
    
    if ps aux | grep -q "labrys_process_guardian" | grep -v grep; then
        echo "✅ Guardian Status: ACTIVE"
        echo ""
        echo "📊 Guardian Activity:"
        if [ -f "labrys_guardian_report.json" ]; then
            echo "   Last report: $(stat -f %Sm labrys_guardian_report.json)"
        fi
        
        echo ""
        echo "🔧 Recent Maintenance:"
        # Show last few lines of any maintenance logs
        if [ -f ".labrys/maintenance.log" ]; then
            tail -5 .labrys/maintenance.log
        else
            echo "   No maintenance actions logged"
        fi
    else
        echo "❌ Guardian Status: NOT RUNNING"
        echo "   Consider starting guardian process"
    fi
    
    echo ""
    echo "⏸️  Next update in 15 seconds..."
    sleep 15
done
