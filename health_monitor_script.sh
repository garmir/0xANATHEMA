
#!/bin/bash
cd "/Users/anam/temp/0xANATHEMA"
source venv/bin/activate

while true; do
    clear
    echo "🗲 LABRYS Health Monitor - $(date)"
    echo "   " + "="*50
    python3 check_labrys_health.py
    echo ""
    echo "⏸️  Next update in 10 seconds..."
    sleep 10
done
