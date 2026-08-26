#!/bin/bash
#
# MlxVoice Release Packaging
#
# Builds a signed Release app, validates its code signature and produces a
# distributable zip. Useful for publishing builds to GitHub Releases or
# sharing with testers.
#
# Usage:
#   ./scripts/package-release.sh                # build + package
#   ./scripts/package-release.sh --no-build     # package existing DerivedData build only
#   ./scripts/package-release.sh --unsigned     # skip signing (CI / fallback)
#
# Output:
#   release/MlxVoice-v<MARKETING_VERSION>.zip
#   release/MlxVoice-v<MARKETING_VERSION>-dSYM.zip   (debug symbols, optional)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_DIR}"

DERIVED_DATA_PATH="${FLUIDVOICE_DERIVED_DATA_PATH:-${PROJECT_DIR}/DerivedData}"
APP_PATH="${DERIVED_DATA_PATH}/Build/Products/Release/MlxVoice.app"
DSYM_PATH="${DERIVED_DATA_PATH}/Build/Products/Release/MlxVoice.app.dSYM"
OUT_DIR="${PROJECT_DIR}/release"
NO_BUILD=0
UNSIGNED=0

for arg in "$@"; do
    case "${arg}" in
        --no-build) NO_BUILD=1 ;;
        --unsigned) UNSIGNED=1 ;;
        *) echo "Unknown option: ${arg}" >&2; exit 1 ;;
    esac
done

resolve_development_team() {
    # Reuse build.sh's logic: take the first valid Apple Development identity.
    security find-identity -v -p codesigning 2>/dev/null \
        | grep -o '"Apple Development:[^"]*"' \
        | head -1 \
        | tr -d '"'
}

build_release() {
    if [ "${UNSIGNED}" = "1" ]; then
        echo "==> Building unsigned Release..."
        xcodebuild -project Fluid.xcodeproj -scheme Fluid \
            -configuration Release -destination 'platform=macOS' \
            -derivedDataPath "${DERIVED_DATA_PATH}" \
            build CODE_SIGNING_ALLOWED=NO CODE_SIGNING_REQUIRED=NO
        return
    fi

    local identity
    identity="$(resolve_development_team)"
    if [ -z "${identity}" ]; then
        echo "!! No Apple Development signing identity found." >&2
        echo "   Install a development certificate or re-run with --unsigned." >&2
        exit 1
    fi

    local team
    team="$(security find-certificate -c "${identity}" -p 2>/dev/null \
        | openssl x509 -noout -subject -nameopt RFC2253 2>/dev/null \
        | sed -n 's/.*OU=\([^,]*\).*/\1/p')"

    echo "==> Building signed Release (team=${team:-unknown})..."
    xcodebuild -project Fluid.xcodeproj -scheme Fluid \
        -configuration Release -destination 'platform=macOS' \
        -derivedDataPath "${DERIVED_DATA_PATH}" \
        build DEVELOPMENT_TEAM="${team}" CODE_SIGN_IDENTITY="Apple Development"
}

package() {
    if [ ! -d "${APP_PATH}" ]; then
        echo "!! Missing $(basename "${APP_PATH}"). Build first or pass --no-build." >&2
        exit 1
    fi

    mkdir -p "${OUT_DIR}"

    if [ "${UNSIGNED}" != "1" ]; then
        echo "==> Validating code signature..."
        codesign --verify --deep --strict "${APP_PATH}" 2>&1 || {
            echo "!! Signature verification failed." >&2
            exit 1
        }
        codesign -dv --verbose=2 "${APP_PATH}" 2>&1 | grep -E "Identifier|TeamIdentifier" || true
    fi

    local version
    version="$(defaults read "${APP_PATH}/Contents/Info" CFBundleShortVersionString 2>/dev/null || echo "dev")"
    local app_zip="${OUT_DIR}/MlxVoice-v${version}.zip"
    local dsym_zip="${OUT_DIR}/MlxVoice-v${version}-dSYM.zip"

    echo "==> Packaging app (v${version})..."
    rm -f "${app_zip}"
    ditto -c -k --keepParent "${APP_PATH}" "${app_zip}"

    if [ -d "${DSYM_PATH}" ]; then
        echo "==> Packaging dSYM..."
        rm -f "${dsym_zip}"
        ditto -c -k --keepParent "${DSYM_PATH}" "${dsym_zip}"
    fi

    echo
    echo "Done:"
    [ -f "${app_zip}" ] && echo "  ${app_zip}  ($(du -h "${app_zip}" | cut -f1))"
    [ -f "${dsym_zip}" ] && echo "  ${dsym_zip}  ($(du -h "${dsym_zip}" | cut -f1))"
    echo
    echo "Publish the app zip to GitHub Releases; users download, unzip and drag"
    echo "MlxVoice.app into /Applications."
}

if [ "${NO_BUILD}" = "1" ]; then
    echo "==> Skipping build (--no-build)"
else
    build_release
fi

package
