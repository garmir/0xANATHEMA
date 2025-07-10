#!/bin/bash
# System Integrity Check - Phase 1 Validation
echo "System Integrity Check Starting..."
echo "Checking Task Master components..."

# Basic validation
if [ -d ".taskmaster" ]; then
    echo "✓ Task Master directory exists"
else
    echo "✗ Task Master directory missing"
    exit 1
fi

if [ -f ".taskmaster/tasks/tasks.json" ]; then
    echo "✓ Tasks configuration exists"
else
    echo "✗ Tasks configuration missing"
fi

if [ -d ".taskmaster/catalytic" ]; then
    echo "✓ Catalytic workspace exists"
else
    echo "✗ Catalytic workspace missing"
fi

if grep -q "pam_tid.so" /etc/pam.d/sudo 2>/dev/null; then
    echo "✓ TouchID PAM configured"
else
    echo "⚠ TouchID PAM configuration needs verification"
fi

echo "System integrity check completed"