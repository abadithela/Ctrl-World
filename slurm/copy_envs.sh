SRC="/n/fs/irom-testing/world_models/Ctrl-World/init_conditions/env_imgs6"
DEST="/n/fs/irom-testing/world_models/Ctrl-World/init_conditions/envs"

# Find the current maximum env number in DEST
max_env=$(
  find "$DEST" -maxdepth 1 -type d -name 'env*' -printf '%f\n' \
  | sed 's/^env//' \
  | sort -n \
  | tail -1
)

# If none exist, start at -1 so first becomes env0
max_env=${max_env:--1}

next_env=$((max_env + 1))

for d in "$SRC"/*; do
    [ -d "$d" ] || continue
    cp -r "$d" "$DEST/env$next_env"
    next_env=$((next_env + 1))
done