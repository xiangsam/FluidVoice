#!/bin/bash
set -euo pipefail

# Fix CTranscribe.framework structure to prevent "Multiple binaries share the same codesign path" error during Xcode Archive.

fix_framework() {
    local fw="$1"
    [ -d "$fw" ] || return 0
    echo "Fixing framework at: $fw"
    
    local versions_dir="${fw}/Versions"
    local version_a="${versions_dir}/A"
    
    if [ -d "${version_a}" ]; then
        # 1. Fix Versions/Current symlink
        if [ ! -L "${versions_dir}/Current" ] || [ "$(readlink "${versions_dir}/Current")" != "A" ]; then
            rm -rf "${versions_dir}/Current"
            (cd "${versions_dir}" && ln -sfn A Current)
        fi
        
        # 2. Fix top-level symlinks
        for item in CTranscribe Headers Modules Resources; do
            if [ -e "${version_a}/${item}" ]; then
                if [ ! -L "${fw}/${item}" ]; then
                    rm -rf "${fw}/${item}"
                    (cd "${fw}" && ln -sfn "Versions/Current/${item}" "${item}")
                fi
            fi
        done
    fi
}

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Search in project local DerivedData and user Xcode DerivedData
find "${PROJECT_DIR}" -name "CTranscribe.framework" -type d 2>/dev/null | while read -r fw; do
    fix_framework "$fw"
done

if [ -d "${HOME}/Library/Developer/Xcode/DerivedData" ]; then
    find "${HOME}/Library/Developer/Xcode/DerivedData" -name "CTranscribe.framework" -type d 2>/dev/null | while read -r fw; do
        fix_framework "$fw"
    done
fi

echo "Framework symlinks successfully verified and fixed."
