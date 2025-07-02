#!/bin/bash

DIR1="$1"
DIR2="$2"

PHASE_FILE="$3"

positions=$(find "$DIR1" -type f -printf "%f\n" | sed -E 's/.*_([0-9]+)_[0-9]+$/\1/' | sort -u)
layers=$(find "$DIR1" -type f -printf "%f\n" | awk -F'_' 'NF >= 3 && $NF ~ /^[0-9]+$/ { print $NF }' | sort -n -u)




COL_WIDTH_PHASE=40
COL_WIDTH=10


printf "%-${COL_WIDTH_PHASE}s %-${COL_WIDTH}s" "Phase" "Layer"
for pos in $positions; do
    printf "%-${COL_WIDTH}s" "$pos"
done
echo


while read -r phase; do
    [ -z "$phase" ] && continue
    for layer in $layers; do
        printf "%-${COL_WIDTH_PHASE}s %-${COL_WIDTH}s" "$phase" "$layer"
        for pos in $positions; do
            file1="$DIR1/${phase}_${pos}_${layer}"
            file2="$DIR2/${phase}_${pos}_${layer}"

            if [[ -f "$file1" && -f "$file2" ]]; then
                if diff -q "$file1" "$file2" > /dev/null; then
                    printf "%-${COL_WIDTH}s" "OK"
                else
                    printf "%-${COL_WIDTH}s" "X"
                fi
            else
                printf "%-${COL_WIDTH}s" "-"
            fi
        done
        echo
    done
done < "$PHASE_FILE"
