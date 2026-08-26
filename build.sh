#!/bin/bash

# MlxVoice Build Profile Router
# Defaults to the public OSS build, which skips private Fluid Intelligence.
#
# Usage:
#   ./build.sh                    # signed public OSS build
#   ./build.sh public             # signed public OSS build
#   ./build.sh unsigned           # unsigned public OSS build (CI/fallback)
#   ./build.sh fi                 # private FI build

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROFILE="${1:-${BUILD_PROFILE:-public}}"
PRIVATE_FI_BUILD_SCRIPT="${PROJECT_DIR}/build_with_FI_incremental.sh"
DERIVED_DATA_PATH="${FLUIDVOICE_DERIVED_DATA_PATH:-${PROJECT_DIR}/DerivedData}"

# Ensure framework symlinks are valid to prevent Xcode codesign errors
if [ -x "${PROJECT_DIR}/scripts/fix-framework-symlinks.sh" ]; then
    "${PROJECT_DIR}/scripts/fix-framework-symlinks.sh" >/dev/null 2>&1 || true
fi

resolve_development_team() {
    local identity
    identity="$(security find-identity -v -p codesigning 2>/dev/null \
        | sed -n 's/.*"\(Apple Development:[^"]*\)".*/\1/p' \
        | awk 'NR == 1 { identity = $0 } END { print identity }')"
    [ -n "${identity}" ] || return 0

    if [ -n "${FLUIDVOICE_DEVELOPMENT_TEAM:-}" ]; then
        printf '%s\n' "${FLUIDVOICE_DEVELOPMENT_TEAM}"
        return
    fi

    security find-certificate -c "${identity}" -p 2>/dev/null \
        | openssl x509 -noout -subject -nameopt RFC2253 2>/dev/null \
        | sed -n 's/.*OU=\([^,]*\).*/\1/p'
}

run_public_build() {
    local signing_mode="$1"
    local config="${2:-Debug}"
    local development_team
    local -a build_args=(
        -project Fluid.xcodeproj
        -scheme Fluid
        -configuration "${config}"
        -destination 'platform=macOS'
        -derivedDataPath "${DERIVED_DATA_PATH}"
        build
    )

    cd "${PROJECT_DIR}"

    if [ "${signing_mode}" = "unsigned" ]; then
        echo "Running unsigned public MlxVoice build (${config})..."
        exec xcodebuild "${build_args[@]}" CODE_SIGNING_ALLOWED=NO CODE_SIGNING_REQUIRED=NO
    fi

    development_team="$(resolve_development_team)"
    if [ -z "${development_team}" ]; then
        if [ -n "${FLUIDVOICE_DEVELOPMENT_TEAM:-}" ]; then
            printf >&2 'FLUIDVOICE_DEVELOPMENT_TEAM is set to %s, but no Apple Development signing identity was found.\n\n' \
                "${FLUIDVOICE_DEVELOPMENT_TEAM}"
        else
            printf >&2 'No Apple Development signing identity was found.\n\n'
        fi
        exit 1
    fi

    echo "Running signed public MlxVoice build (${config})..."
    exec xcodebuild "${build_args[@]}" DEVELOPMENT_TEAM="${development_team}"
}

case "${PROFILE}" in
    public|oss|incremental|fast)
        run_public_build signed Debug
        ;;
    release|prod)
        run_public_build unsigned Release
        ;;
    unsigned|ci)
        run_public_build unsigned Debug
        ;;
    fi|private|dev|full)
        if [ ! -x "${PRIVATE_FI_BUILD_SCRIPT}" ]; then
            echo "Private Fluid Intelligence build script is missing:"
            echo "  ${PRIVATE_FI_BUILD_SCRIPT}"
            echo "Restore the private FI build setup, then run: sh build_with_FI_incremental.sh"
            exit 1
        fi
        exec "${PRIVATE_FI_BUILD_SCRIPT}"
        ;;
    *)
        echo "Unknown build profile: ${PROFILE}"
        echo "Valid profiles: public/oss/incremental/fast, unsigned/ci, fi/private/dev/full"
        exit 1
        ;;
esac
