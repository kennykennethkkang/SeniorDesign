#!/bin/bash

append_ld_library_path() {
  local candidate
  local new_paths=""
  local separator=""
  for candidate in "$@"; do
    if [ -d "$candidate" ]; then
      case ":${LD_LIBRARY_PATH:-}:" in
        *":$candidate:"*)
          ;;
        *)
          case ":$new_paths:" in
            *":$candidate:"*)
              ;;
            *)
              new_paths="${new_paths}${separator}${candidate}"
              separator=":"
              ;;
          esac
          ;;
      esac
    fi
  done
  if [ -n "$new_paths" ]; then
    LD_LIBRARY_PATH="$new_paths${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  export LD_LIBRARY_PATH
}

expose_venv_native_libraries() {
  local site_packages
  local native_dirs=()
  local nvidia_lib_dir
  site_packages="$(
    python - <<'PY'
import sysconfig

print(sysconfig.get_paths().get("platlib") or "")
PY
  )"

  if [ -z "$site_packages" ]; then
    return
  fi

  native_dirs+=(
    "$site_packages/ctranslate2.libs"
    "$site_packages/torch/lib"
  )

  for nvidia_lib_dir in "$site_packages"/nvidia/*/lib; do
    case "$nvidia_lib_dir" in
      "$site_packages/nvidia/cu13/lib")
        ;;
      *)
        native_dirs+=("$nvidia_lib_dir")
        ;;
    esac
  done

  append_ld_library_path "${native_dirs[@]}"
}
