#!/bin/bash
# Clear Omniverse and USD cache to fix material issues

echo "🧹 Clearing Omniverse/USD caches..."

# Clear Omniverse pycache
if [ -d ~/.nvidia-omniverse/pycache ]; then
    echo "  Clearing ~/.nvidia-omniverse/pycache..."
    rm -rf ~/.nvidia-omniverse/pycache/*
    echo "  ✅ Cleared Omniverse pycache"
fi

# Clear Omniverse logs (optional, can help with debugging)
if [ -d ~/.nvidia-omniverse/logs ]; then
    echo "  Clearing ~/.nvidia-omniverse/logs..."
    rm -rf ~/.nvidia-omniverse/logs/*
    echo "  ✅ Cleared Omniverse logs"
fi

# Clear Python cache files
echo "  Clearing Python __pycache__ directories..."
find /home/ununtu/code -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
echo "  ✅ Cleared Python cache"

# Clear Genesis USD bake cache (this is the most common issue!)
if [ -d ~/.cache/genesis/usd ]; then
    echo "  Clearing Genesis USD bake cache..."
    CACHE_SIZE=$(du -sh ~/.cache/genesis/usd 2>/dev/null | cut -f1)
    echo "    Cache size: $CACHE_SIZE"
    rm -rf ~/.cache/genesis/usd/
    echo "  ✅ Cleared Genesis USD cache"
fi

# Clear entire Genesis cache if needed
if [ "$1" == "--all" ]; then
    if [ -d ~/.cache/genesis ]; then
        echo "  Clearing entire Genesis cache..."
        CACHE_SIZE=$(du -sh ~/.cache/genesis 2>/dev/null | cut -f1)
        echo "    Cache size: $CACHE_SIZE"
        rm -rf ~/.cache/genesis/
        echo "  ✅ Cleared entire Genesis cache"
    fi
fi

# Clear pip cache (USD related packages)
echo "  Clearing pip cache for USD packages..."
pip cache remove '*usd*' 2>/dev/null
echo "  ✅ Cleared pip USD cache"

# Clear system temp files
echo "  Clearing /tmp USD/Omni related files..."
find /tmp -maxdepth 1 -type f -o -type d \( -name "*omni*" -o -name "*usd*" -o -name "*isaac*" \) -mtime +1 -exec rm -rf {} + 2>/dev/null
echo "  ✅ Cleared temp files"

echo ""
echo "✅ Cache clearing complete!"
echo ""
echo "📝 Next steps:"
echo "  1. Restart your terminal/IDE"
echo "  2. Re-run your USD import test"
echo "  3. If issue persists, check USD file itself with:"
echo "     usdview /path/to/your/file.usd"
echo "  4. Or use usdcat to inspect materials:"
echo "     usdcat /path/to/your/file.usd | grep -A 10 'def Material'"
